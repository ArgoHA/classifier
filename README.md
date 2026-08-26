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
- **train.decision_metrics** — mean of these picks the best checkpoint
- **export.formats** / **bench.formats** — `null` for all, or a list to restrict
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
label, prob = model(cv2.imread("img.jpg"))   # BGR in, as cv2 hands it over
```

Bare `state_dict` checkpoints from before the envelope still load: missing facts are
recovered from the `config.yaml` that training freezes next to the weights.

Available wrappers: `TorchModel`, `TRTModel`, `OVModel`, `ONNXModel`. All take BGR.

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
+----------+-------------+------------+----------------+
|  format  | mean_cosine | min_cosine | top1_agreement |
+----------+-------------+------------+----------------+
|   ONNX   |     1.0     |  0.999999  |      8/8       |
| OpenVINO |     1.0     |  0.999999  |      8/8       |
| TensorRT |     1.0     |  0.999998  |      8/8       |
+----------+-------------+------------+----------------+
```

Cosine of the whole softmax vector, not just the argmax: drift shows up there before it
changes an answer. Below 0.9999 (fp32) it warns loudly.

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
