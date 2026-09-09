"""Export a trained checkpoint to ONNX / OpenVINO / TensorRT, then prove the graphs match.

An export that produces a file is not an export that works. Everything here after the
writers exists because a silently-wrong graph is worse than a failed build: the class-count
guard refuses to export a head the checkpoint never trained, and the parity check compares
each backend's full softmax against torch on real validation images.

Backends are imported lazily and gated by `export.formats`, so a missing openvino cannot
break an ONNX-only export, and the research loop can rebuild one backend without paying
for the others.
"""

from pathlib import Path
from typing import List, Optional, Sequence

import hydra
import numpy as np
import pandas as pd
import torch
from loguru import logger
from omegaconf import DictConfig
from tabulate import tabulate
from torch import nn

from img_clf.config.resolve import CONFIG_NAME, config_dir
from img_clf.dl.bench import load_backends
from img_clf.dl.ckpt import describe_artifact, norm_kwargs
from img_clf.dl.model import prepare_model
from img_clf.dl.utils import get_latest_experiment_name, resolve_formats

# Order is the order they build in: OpenVINO and TensorRT both consume the ONNX graph.
# torch is the source of an export, not one of its products, so it is not here.
EXPORT_FORMATS = ("onnx", "openvino", "tensorrt")


def _check_class_count(model_path: Path, n_ckpt: Optional[int], n_config: int) -> None:
    """Refuse to export a checkpoint whose head disagrees with `train.label_to_name`.

    `load_state_dict` would raise here, but only after the export dir has been half
    written; more importantly the same mismatch under a non-strict load silently drops the
    head and exports it at random init, and parity then compares that corrupt model
    against graphs made from it - so the export "passes".
    """
    if n_ckpt is not None and n_ckpt != n_config:
        raise ValueError(
            f"{model_path.name} has a {n_ckpt}-class head but train.label_to_name lists "
            f"{n_config}. Point the config at the classes this checkpoint was trained on, "
            "or retrain before exporting."
        )


def export_to_onnx(
    model: nn.Module,
    model_path: Path,
    x_test: torch.Tensor,
    max_batch_size: int,
    half: bool,
    dynamic_input: bool,
    input_name: str,
    output_names: List[str],
    simplify: bool = False,
) -> Path:
    """Write `model_path.with_suffix(".onnx")` and return it.

    fp16 is done by exporting a halved module rather than converting the graph afterwards:
    the wrappers feed the graph `np.float16` when `export.half` is set, so the graph's own
    input must be fp16 too.
    """
    import onnx

    if half:
        model = model.half()
        x_test = x_test.half()

    dynamic_axes = {}
    if max_batch_size > 1:
        for name in [input_name] + output_names:
            dynamic_axes[name] = {0: "batch_size"}
    if dynamic_input:
        dynamic_axes.setdefault(input_name, {}).update({2: "height", 3: "width"})

    output_path = model_path.with_suffix(".onnx")
    torch.onnx.export(
        model,
        x_test,
        opset_version=19,
        input_names=[input_name],
        output_names=output_names,
        dynamic_axes=dynamic_axes if dynamic_axes else None,
        dynamo=True,
    ).save(output_path)
    logger.info("ONNX model exported")

    if simplify:
        import onnxsim

        try:
            simplified, check = onnxsim.simplify(onnx.load(output_path))
            assert check, "onnxsim reported the simplified graph is not equivalent"
        except Exception as e:
            logger.info(f"Simplification failed, keeping the exported graph: {e}")
        else:
            onnx.save(simplified, output_path)
            logger.info("ONNX simplified and exported")
    return output_path


def export_to_openvino(
    onnx_path: Path,
    x_test: torch.Tensor,
    dynamic_input: bool,
    max_batch_size: int,
    input_name: str,
    output_name: str,
) -> None:
    """Converted from the ONNX graph, not from the live nn.Module.

    Both work, but going through ONNX means OpenVINO and TensorRT consume identical
    input, so a parity disagreement points at one runtime rather than at two independent
    conversion paths.

    `input` is left as None for the static 1-image case so OpenVINO keeps the shape the
    ONNX graph already carries; -1 marks whichever of batch / height / width is free.
    """
    import openvino as ov

    channels = int(x_test.shape[1])
    inp = None
    if max_batch_size > 1 and dynamic_input:
        inp = [-1, channels, -1, -1]
    elif max_batch_size > 1:
        inp = [-1, *x_test.shape[1:]]
    elif dynamic_input:
        inp = [1, channels, -1, -1]

    model = ov.convert_model(input_model=str(onnx_path), input=inp, example_input=x_test)
    model.inputs[0].tensor.set_names({input_name})
    model.outputs[0].tensor.set_names({output_name})
    ov.serialize(model, str(onnx_path.with_suffix(".xml")), str(onnx_path.with_suffix(".bin")))
    logger.info("OpenVINO model exported")


def export_to_tensorrt(
    onnx_file_path: Path, half: bool, max_batch_size: int, opt_bs: int = 1
) -> None:
    import onnx
    import tensorrt as trt

    opt_bs = min(opt_bs, max_batch_size)

    tr_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(tr_logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, tr_logger)

    with open(onnx_file_path, "rb") as model:
        if not parser.parse(model.read()):
            errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError("Failed to parse the ONNX file:\n" + "\n".join(errors))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2GB
    if half:
        config.set_flag(trt.BuilderFlag.FP16)

    if max_batch_size > 1:
        profile = builder.create_optimization_profile()
        input_name = network.get_input(0).name

        # Read the non-batch dims off the ONNX graph rather than the parsed network: the
        # parsed tensor reports the free batch dim as -1, and the graph is the thing that
        # actually says whether height/width are static.
        onnx_model = onnx.load(str(onnx_file_path))
        shape_proto = next(
            (i.type.tensor_type.shape for i in onnx_model.graph.input if i.name == input_name),
            None,
        )
        if shape_proto is None:
            raise ValueError(
                f"Could not find input '{input_name}' in the ONNX graph. Available: "
                f"{[i.name for i in onnx_model.graph.input]}"
            )

        static_dims = []
        for i, dim in enumerate(shape_proto.dim[1:], start=1):  # skip batch
            if not dim.dim_value:
                raise ValueError(
                    f"input has a dynamic dimension at index {i}; only batch may be dynamic "
                    "in a TensorRT optimization profile"
                )
            static_dims.append(int(dim.dim_value))

        profile.set_shape(
            input_name,
            (1, *static_dims),
            (opt_bs, *static_dims),
            (max_batch_size, *static_dims),
        )
        config.add_optimization_profile(profile)

    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError(
            "Failed to build TensorRT engine. Usual causes: not enough GPU memory, an "
            "unsupported op in the ONNX graph, or a bad dynamic-batch profile. The "
            "TensorRT log above has the specifics."
        )

    with open(onnx_file_path.with_suffix(".engine"), "wb") as f:
        f.write(engine)
    logger.info("TensorRT model exported")


def _parity_images(cfg, n: int) -> List[np.ndarray]:
    """Real validation images. Random noise has no confident prediction, so a graph that
    quietly broke one class would still score a high cosine on it."""
    import cv2

    data_path = Path(cfg.train.data_path)
    csv_path = data_path / "val.csv"
    if not csv_path.is_file():
        logger.warning(f"{csv_path} not found; parity will use random input")
        h, w = cfg.train.img_size
        return [np.random.randint(0, 255, (h, w, 3), dtype=np.uint8) for _ in range(n)]

    split = pd.read_csv(csv_path, header=None)
    # spread across the file so several classes are represented, not just the first one
    step = max(1, len(split) // n)
    images = []
    for idx in range(0, len(split), step):
        if len(images) >= n:
            break
        img = cv2.imread(str(data_path / split.iloc[idx, 0]))
        if img is not None:
            images.append(img)
    return images


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return float(a @ b / denom)


def run_parity(cfg, models_dir: Path, selected: Sequence[str], n_images: int, half: bool) -> None:
    """Compare every exported backend against torch on real images, singly and as a batch.

    Per backend: cosine of the full probability vector, which catches drift before it
    changes an answer, and top-1 agreement, which is what actually ships - once for
    single-image calls and once for one `probs` call over the whole sample, each against
    the same torch call. Plus the largest gap between the backend's own batched and
    single-image rows: a batched forward may pick other kernels, but it must not change an
    answer, so that gap has to be noise.
    """
    from img_clf.infer.torch_model import TorchModel

    images = _parity_images(cfg, n_images)
    if not images:
        logger.warning("Parity: no images available to compare on")
        return
    n = len(images)

    # The same normalization the graph wrappers get. Without it the reference preprocesses
    # differently from everything it is compared against, and a byte-identical export reads
    # as a parity failure - or, worse, coincidentally passes and hides a real one.
    norms = norm_kwargs(models_dir)
    reference = TorchModel(model_path=str(models_dir / "model.pt"), half=half, **norms)
    ref_single = np.concatenate([reference.probs(img) for img in images])
    ref_batch = reference.probs(images)

    # fp32 must round-trip almost exactly; fp16 legitimately loses a little.
    threshold = 0.99 if half else 0.9999
    # A backend's batched and single-image rows may come off different kernels: cuDNN's
    # default TF32 convolutions (torch, ORT) put them up to ~3.5e-3 apart on efficientnet_b0
    # probabilities; TensorRT happens to be exact. 1e-2 leaves margin and still catches a
    # swapped row or a softmax over the wrong axis, which differ by ~1.
    batch_atol = 5e-2 if half else 1e-2
    ref_gap = float(np.abs(ref_batch - ref_single).max())
    if ref_gap > batch_atol:
        logger.warning(
            f"torch: batched rows differ from single-image rows by up to {ref_gap:.1e} "
            f"(> {batch_atol:g}); the reference itself is not batch-stable"
        )

    rows, failures = [], []
    # Same loader bench uses, so the two reports name the backends identically and one
    # graph is resident at a time. torch is the reference above, never a row here.
    for tag, model in load_backends(models_dir, selected, half):
        single = np.concatenate([model.probs(img) for img in images])
        batch = model.probs(images)

        cosines = [_cosine(r, o) for r, o in zip(ref_single, single)]
        top1 = int((single.argmax(1) == ref_single.argmax(1)).sum())
        batch_cosines = [_cosine(r, o) for r, o in zip(ref_batch, batch)]
        batch_top1 = int((batch.argmax(1) == ref_batch.argmax(1)).sum())
        gap = float(np.abs(batch - single).max())

        mean_cos, min_batch_cos = float(np.mean(cosines)), float(min(batch_cosines))
        rows.append(
            [
                tag,
                round(mean_cos, 6),
                round(min(cosines), 6),
                f"{top1}/{n}",
                round(min_batch_cos, 6),
                f"{batch_top1}/{n}",
                f"{gap:.1e}",
            ]
        )
        if mean_cos < threshold or top1 < n:
            failures.append(f"{tag} single: cos={mean_cos:.6f} top1={top1}/{n}")
        if min_batch_cos < threshold or batch_top1 < n:
            failures.append(f"{tag} batch: min cos={min_batch_cos:.6f} top1={batch_top1}/{n}")
        if gap > batch_atol:
            failures.append(f"{tag} batch-vs-single gap {gap:.1e} > {batch_atol:g}")

    if not rows:
        logger.info("Parity: no exported backends available to compare against torch")
        return

    headers = [
        "format",
        "mean_cosine",
        "min_cosine",
        "top1_agreement",
        "batch_min_cosine",
        "batch_top1_agreement",
        "batch_vs_single",
    ]
    pd.DataFrame(rows, columns=headers).to_csv(models_dir / "parity.csv", index=False)
    print("\n" + tabulate(rows, headers=headers, tablefmt="pretty"))
    if failures:
        logger.warning(
            f"Parity below threshold (cosine < {threshold}, a top-1 disagreement, or a "
            f"batch-vs-single gap > {batch_atol:g}) vs torch: " + "; ".join(failures)
        )
    else:
        logger.info(
            f"Parity OK: every backend matches torch singly and batched (cosine >= "
            f"{threshold}, top-1 {n}/{n}, batch-vs-single gap <= {batch_atol:g})"
        )


@hydra.main(version_base=None, config_path=config_dir(), config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    input_name = "input"
    output_name = "output"

    cfg.exp = get_latest_experiment_name(cfg.exp, cfg.train.path_to_save)
    models_dir = Path(cfg.train.path_to_save)
    model_path = models_dir / "model.pt"
    if not model_path.is_file():
        raise FileNotFoundError(f"{model_path} not found - train before exporting.")

    num_classes = len(cfg.train.label_to_name)
    _check_class_count(model_path, describe_artifact(model_path)["num_classes"], num_classes)

    selected = resolve_formats(cfg.export.formats, EXPORT_FORMATS, "export.formats")
    logger.info(f"Exporting: {', '.join(selected)}")

    half = cfg.export.half
    max_batch_size = cfg.export.max_batch_size
    dynamic_input = cfg.export.dynamic_input

    model = prepare_model(cfg.model_name, model_path, num_classes, cfg.train.device)
    x_test = torch.randn(max_batch_size, 3, *cfg.train.img_size).to(cfg.train.device)
    _ = model(x_test)  # fail here, on a shape mismatch, rather than inside a converter
    if half:
        x_test = x_test.half()

    # onnx is needed for both openvino and tensorrt
    onnx_path = export_to_onnx(
        model,
        model_path,
        x_test,
        max_batch_size,
        half=half,
        dynamic_input=dynamic_input,
        input_name=input_name,
        output_names=[output_name],
        simplify=cfg.export.simplify,
    )

    if "openvino" in selected:
        export_to_openvino(
            onnx_path,
            x_test,
            dynamic_input,
            max_batch_size,
            input_name=input_name,
            output_name=output_name,
        )

    if "tensorrt" in selected:
        export_to_tensorrt(onnx_path, half, max_batch_size, opt_bs=cfg.export.opt_batch_size or 1)

    logger.info(f"Exports saved to: {models_dir}")

    if cfg.export.parity:
        run_parity(cfg, models_dir, selected, int(cfg.export.parity_images), half)


if __name__ == "__main__":
    main()
