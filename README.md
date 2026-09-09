# Classifier

A pipeline for training, exporting and benchmarking image classification models with PyTorch
and [timm](https://huggingface.co/docs/timm/en/models/). Hydra configs, optional WandB tracking,
and exports to ONNX, TensorRT and OpenVINO with a parity check against torch.

## Install

Managed with [uv](https://docs.astral.sh/uv/). `uv sync` installs everything including the
export backends and TensorRT:

```bash
uv sync                      # full clone workflow
uv sync --no-group dev       # runtime only
```

Every entrypoint is a plain module with its own Hydra `main()`. Use the make targets, or call
the module directly when you want overrides:

```bash
make train
uv run python -m img_clf.dl.train model_name=resnet50 train.epochs=100
uv run python -m img_clf.dl.export export.formats=[onnx]
make train ARGS="train.epochs=50"      # overrides through make
```

Dependency notes worth knowing before you bump anything:

- `opencv-python-headless` is **pinned**, not floored. 4.14 made small-input `INTER_AREA`
  resize ~10% slower, which is a measurable fraction of single-image latency.
- `onnxruntime-gpu>=1.29` is the first CUDA 13 build, so it reuses the CUDA libs torch's
  cu130 wheels already ship. Older (cu12) builds pull `nvidia-cudnn-cu12`, which overwrites
  torch's bundled cuDNN in a shared directory and breaks torch.

## Configuration

`config.yaml` at the repo root is the live config; `img_clf/config/default.yaml` is a
sanitized template to copy from. Key fields:

- **model_name** — any timm model name
- **train.root** — project root holding the dataset and receiving all outputs
- **train.data_path** — dataset dir, one subfolder per class
- **train.label_to_name** — class index -> folder name
- **train.amp_dtype** — `bfloat16` (default) or `float16`
- **train.batch_size** — physical batch per step; `-1` probes the GPU for the largest batch
  whose training step (forward, backward, AdamW update) fits in 70% of VRAM. CUDA only,
  takes a few seconds before the first epoch
- **train.decision_metrics** — mean of these picks the best checkpoint
- **export.formats** / **bench.formats** — `null` for all, or a list to restrict
- **export.max_batch_size** / **export.opt_batch_size** — batch axis of the exported graphs;
  `1` (default) bakes in batch 1, see [Batched inference](#batched-inference)
- **exp** — experiment name used for output paths across train/export/bench

Normalization is **not** configured: it comes from the model's timm `pretrained_cfg`, so
inception- and xception-style models get their 0.5/0.5 stats instead of ImageNet's.

## Usage

```bash
make preprocess      # convert images and PDFs to jpg
make split           # write train/val/test csvs
make train           # train
make export          # export to onnx / tensorrt / openvino + parity check
make bench           # run every exported backend over the test set
make infer           # inference over a folder
make vis             # Grad-CAM heatmaps
make check_errors    # dump misclassified images by confusion pair
make test            # pytest
```

`make main` runs train -> export -> bench in sequence.

## Checkpoints describe themselves

Training writes `{"model": state_dict, "meta": {...}}`, where meta carries `model_name`,
`num_classes`, `label_to_name`, `img_size`, `mean` and `std`. Every wrapper therefore needs
only a path:

```python
from img_clf.infer.trt_model import TRTModel

model = TRTModel(model_path="output/models/exp/model.engine")
pred = model(cv2.imread("img.jpg"))[0]   # BGR in, as cv2 hands it over
pred["label"], pred["score"]             # 3, 0.97 - the class id and its probability
pred["class_name"], pred["probs"]        # its name, and every class's probability by name
```

`label` is always the class id; `class_name` and the `probs` keys are names when the
wrapper knows them. `TorchModel` reads `label_to_name` out of the envelope, so it names
classes on its own. The graph wrappers stay self-contained - they never import the
checkpoint code - so they take `label_to_name={0: "excavator", ...}` from whoever builds
them, and stringify the id for classes left unnamed.

Bare `state_dict` checkpoints from before the envelope still load: missing facts are
recovered from the `config.yaml` that training freezes next to the weights.

Available wrappers: `TorchModel`, `TRTModel`, `OVModel`, `ONNXModel`, `ZeroShotModel`.
All take BGR.

### Zero-shot, no training

`ZeroShotModel` classifies with class names instead of a trained checkpoint: an open_clip
dual encoder (SigLIP 2; the image tower is a timm model) scores each crop against one text
prompt per label. Same contract as `TorchModel`; ids follow `train.label_to_name`, while
normalization and resize mode come from the checkpoint - swapping `infer.hub` swaps the
preprocessing with it (SigLIP 2 squashes, CLIP scales the short edge and center-crops). `make infer ARGS="infer.zero_shot=true"` runs it;
`infer.hub` picks the checkpoint, `infer.template` the prompt.

```python
from img_clf.infer.zero_shot_model import ZeroShotModel

model = ZeroShotModel(hub="hf-hub:timm/ViT-SO400M-16-SigLIP2-256", labels={0: "car", 1: "bus"})
model.probs(img, labels=["car", "bus"])   # per-call label sets work too
```

Needs the `zero_shot` extra (in `[all]`; `transformers` rides along only for the HF
tokenizer).

### Batched inference

Every wrapper also classifies a batch - "N crops out of one frame, one forward pass":

```python
probs = model.probs(images)      # (N, C) float32 softmax rows, row i for images[i]
preds = model(images)            # [{"label": class_id, "class_name", "score", "probs"}, ...]
model.max_batch_size             # int, or None when the graph has no batch limit
```

`probs` and `__call__` take one BGR image or a sequence of them (list, tuple, or an
N x H x W x 3 array); a lone image counts as N = 1. Order is preserved, sequences longer
than `max_batch_size` are chunked internally, and empty input raises `ValueError`. A graph
exported at batch 1 still works - a sequence degrades to a per-image loop.

`max_batch_size` comes off the graph at load:

| wrapper | `max_batch_size` |
|---|---|
| `TorchModel` | `None` - an `nn.Module` takes any batch |
| `TRTModel` | the engine profile's max batch (`get_tensor_profile_shape`); a static batch-1 engine reports `1` |
| `ONNXModel` | `None` when the batch axis is dynamic, else the baked-in size (`1`) |
| `OVModel` | `None` when the batch axis is free, else `1` - it recompiles at batch 1 when the device cannot run a free axis |

#### Exporting for a batched service

```yaml
export:
  max_batch_size: 32     # dynamic batch axis; TensorRT profile max
  opt_batch_size: 8      # batch TensorRT tunes its kernels for
  dynamic_input: False   # H/W must stay static for the TensorRT profile
```

or `make export ARGS="export.max_batch_size=32 export.opt_batch_size=8"`. This writes
`model.onnx` with a `batch_size` axis, `model.engine` with the profile
`(1,3,H,W) / (8,3,H,W) / (32,3,H,W)`, and `model.xml` with batch `-1`; the wrappers then
report `max_batch_size` 32 (TensorRT) / `None` (ONNX; OpenVINO too, or `1` where the device
cannot run a free axis - NVIDIA over OpenCL is one such). The default stays at 1;
`opt_batch_size` only tunes TensorRT's kernels, so keep it at 1 if most requests carry one
crop.

`trt_model.py`, `onnx_model.py` and `ov_model.py` import **nothing** from `img_clf` - copy one
into a service and it works on its own. Each therefore carries its own copy of the
preprocessing; `tests/test_preprocess.py` fails if the copies drift, and asserts the three stay
package-free. `torch_model.py` is the exception, and has to be: a `.pt` is only weights, so
loading one needs timm and the checkpoint reader.

A graph wrapper reads its input size and class count off its own graph. Normalization it cannot
know - that belongs to the training run - so it defaults to ImageNet stats, and everything
driving a run directory passes the trained values in (`ckpt.norm_kwargs`).

## Export parity

`make export` compares every exported backend against torch on real validation images and
writes `parity.csv`:

```
+----------+-------------+------------+----------------+------------------+----------------------+-----------------+
|  format  | mean_cosine | min_cosine | top1_agreement | batch_min_cosine | batch_top1_agreement | batch_vs_single |
+----------+-------------+------------+----------------+------------------+----------------------+-----------------+
|   ONNX   |     1.0     |    1.0     |      8/8       |     0.999998     |         8/8          |     6.8e-04     |
| OpenVINO |     1.0     |    1.0     |      8/8       |       1.0        |         8/8          |     0.0e+00     |
| TensorRT |     1.0     |    1.0     |      8/8       |     0.999999     |         8/8          |     0.0e+00     |
+----------+-------------+------------+----------------+------------------+----------------------+-----------------+
```

Cosine of the whole softmax vector, not just the argmax: drift shows up there before it
changes an answer. The first three columns compare single-image calls against torch; the
`batch_*` columns compare one batched `probs` call against torch; `batch_vs_single` is the
largest gap between the backend's own batched and single-image rows - a batched forward may
pick different kernels, but it must not change the answer. It warns below 0.9999 cosine
(fp32; 0.99 fp16), on any top-1 disagreement, or on a batch-vs-single gap above 1e-2
(5e-2 fp16).

## Outputs

Under `train.root/output/`:

- **models/`exp`/** — `model.pt`, `last.pt`, exported graphs, the resolved run config,
  `metrics.csv`, `extended_metrics.csv`, `parity.csv`, `bench_metrics.csv`, confusion
  matrices, PR curves, `train_log.txt`
- **debug_images/** — images exactly as fed to the model, post-augmentation
- **eval_preds/** — val predictions drawn on the images (GT green, pred blue)
- **bench_imgs/`backend`/`gt`_as_`pred`/** — every misclassified test image, foldered by
  confusion pair, so the dominant confusion is visible at a glance
- **visualized/** — Grad-CAM heatmaps

## Integration example

Input contract for an exported graph:

```
Input tensor: [batch, 3, H, W], float32, RGB, NCHW
Resize:       cv2.INTER_AREA to (W, H)
Scale:        /255, then (x - mean) / std from the checkpoint's meta
```

```python
image = cv2.imread("img.jpg")                      # BGR
image = cv2.resize(image, (384, 384), interpolation=cv2.INTER_AREA)
image = image[:, :, ::-1].transpose(2, 0, 1)       # BGR->RGB, HWC->CHW
image = image.astype(np.float32) / 255.0
image = (image - mean[:, None, None]) / std[:, None, None]
image = image[None]
```

Or just call the wrapper, which does exactly this.
