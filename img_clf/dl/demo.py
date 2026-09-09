"""Image classifier Gradio demo - SigLIP 2 zero-shot (default) or a finetuned checkpoint.

    make demo        (or: uv run python -m img_clf.dl.demo)

Zero-shot scores the upload against comma-separated labels with the config's `infer.hub`
checkpoint (downloaded on first use). Finetuned lists the experiment dirs under
`train.path_to_save` - every model.pt describes itself, so a dropdown entry is all a
checkpoint needs (a pasted path to any model.pt works too).
"""

import time
import warnings
from pathlib import Path

import cv2
import gradio as gr

from img_clf.config.resolve import CONFIG_NAME, config_dir
from img_clf.infer.torch_model import TorchModel
from img_clf.infer.zero_shot_model import ZeroShotModel

HUB_CHOICES = [
    "hf-hub:timm/ViT-SO400M-16-SigLIP2-256",  # the config's default
    "hf-hub:timm/ViT-B-16-SigLIP2-256",
    "hf-hub:timm/ViT-B-32-SigLIP2-256",  # smallest
]
DEFAULT_LABELS = "car, bus, person"
ZERO_SHOT, FINETUNED = "zero-shot (SigLIP 2)", "finetuned"

_MODELS: dict = {}  # ("zs", hub, template) / ("ft", path) -> wrapper; labels ride in per call


def _cfg():
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(config_dir()), version_base=None):
        return compose(config_name=CONFIG_NAME)


def list_checkpoints(cfg) -> list:
    """Experiment dirs under output/models carrying a model.pt, oldest first."""
    models_root = Path(cfg.train.path_to_save).parent
    return sorted(p.parent.name for p in models_root.glob("*/model.pt"))


def _resolve_ckpt(name: str, cfg) -> Path:
    """A dropdown entry resolves under output/models; a pasted path passes through."""
    p = Path(name)
    if not p.is_file():
        p = Path(cfg.train.path_to_save).parent / name / "model.pt"
    if not p.is_file():
        raise gr.Error(f"No model.pt at {p}")
    return p


def _parse_labels(text: str) -> list:
    labels = [t.strip() for t in (text or "").split(",") if t.strip()]
    if len(labels) < 2:
        raise gr.Error("Zero-shot needs at least two comma-separated labels")
    return labels


def _get_model(mode: str, hub: str, template: str, ckpt: str, labels: list, cfg):
    """Zero-shot caches per hub+template (labels ride in per call through the wrapper's own
    cache), finetuned per checkpoint path - so only the first click pays for a load."""
    if mode == ZERO_SHOT:
        key = ("zs", hub, template)
        if key not in _MODELS:
            gr.Info(f"Loading {hub.rsplit('/', 1)[-1]} - first use downloads the weights")
            _MODELS[key] = ZeroShotModel(hub=hub, labels=labels, template=template)
        return _MODELS[key]
    key = ("ft", str(_resolve_ckpt(ckpt, cfg)))
    if key not in _MODELS:
        _MODELS[key] = TorchModel(model_path=key[1])
    return _MODELS[key]


def classify(img, mode: str, hub: str, labels_text: str, template: str, ckpt: str):
    """One upload -> {class: prob} for gr.Label, plus a status line."""
    if img is None:
        raise gr.Error("Upload an image first")
    labels = _parse_labels(labels_text) if mode == ZERO_SHOT else None
    model = _get_model(mode, hub, template, ckpt, labels, _cfg())

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # wrappers take BGR
    t0 = time.perf_counter()
    probs = model.probs(img_bgr, labels=labels) if labels else model.probs(img_bgr)
    ms = (time.perf_counter() - t0) * 1000
    print(f"[demo] {mode} {ms:.1f} ms")

    if labels:  # zero-shot: the columns follow the requested labels, not the load-time ones
        source, scores = hub.rsplit("/", 1)[-1], dict(zip(labels, map(float, probs[0])))
    else:
        names = model.label_to_name or {}
        source = Path(model.model_path).parent.name
        scores = {names.get(i, str(i)): float(p) for i, p in enumerate(probs[0])}
    status = f"✅ {type(model).__name__} | {source} | device: {model.device} | {ms:.1f} ms"
    return scores, status


def build_ui() -> gr.Blocks:
    cfg = _cfg()
    with gr.Blocks(title="img-clf demo") as demo:
        gr.Markdown(
            "# Image classifier demo\nSigLIP 2 zero-shot by default; switch to finetuned to "
            "run a trained experiment."
        )
        with gr.Row():
            with gr.Column():
                mode = gr.Radio([ZERO_SHOT, FINETUNED], value=ZERO_SHOT, label="Model")
                hub = gr.Dropdown(
                    HUB_CHOICES, value=cfg.infer.hub, label="Zero-shot checkpoint",
                    info="first use downloads the weights",
                )
                labels = gr.Textbox(
                    DEFAULT_LABELS, label="Labels",
                    info="comma-separated; each becomes template.format(label)",
                )
                template = gr.Textbox(cfg.infer.template, label="Prompt template")
                ckpt = gr.Dropdown(
                    list_checkpoints(cfg), label="Finetuned checkpoint",
                    info=f"experiments under {Path(cfg.train.path_to_save).parent}; paste a "
                    "model.pt path for anything else",
                    allow_custom_value=True, visible=False,
                )
                img = gr.Image(sources=["upload", "webcam"], type="numpy", label="Image")
                run = gr.Button("Run", variant="primary")
            with gr.Column():
                out = gr.Label(label="Probabilities", num_top_classes=10)
                status = gr.Markdown()

        def toggle(name: str):
            zs = name == ZERO_SHOT
            return [gr.update(visible=zs)] * 3 + [gr.update(visible=not zs)]

        mode.change(toggle, mode, [hub, labels, template, ckpt])
        run.click(classify, [img, mode, hub, labels, template, ckpt], [out, status])
    return demo


def main(host: str = "0.0.0.0", port: int = 7860, share: bool = False) -> None:
    # gradio 6.16 trips this inside its own queue route, once per request
    warnings.filterwarnings("ignore", "'HTTP_422_UNPROCESSABLE_ENTITY' is deprecated")
    if host not in ("127.0.0.1", "localhost"):
        # the checkpoint field loads any model.pt the browser names - LAN access hands
        # that file-loading power to everyone who can reach the port
        print(
            f"WARNING: serving on {host} - anyone who can reach this port can load "
            "any model.pt on this machine"
        )
    # 0.0.0.0 is a bind address, not a thing a browser can open.
    click = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    print(f"Open http://{click}:{port} in your browser")
    build_ui().launch(server_name=host, server_port=port, share=share)


if __name__ == "__main__":
    main()
