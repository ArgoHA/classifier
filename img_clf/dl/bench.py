import gc
import time
from pathlib import Path
from shutil import rmtree
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import cv2
import hydra
import numpy as np
import pandas as pd
import torch
from loguru import logger
from omegaconf import DictConfig
from tabulate import tabulate
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from img_clf.config.resolve import CONFIG_NAME, config_dir
from img_clf.dl.ckpt import norm_kwargs
from img_clf.dl.utils import get_latest_experiment_name, resolve_formats, vis_one_image
from img_clf.dl.validator import Validator

# key -> (row label, artifact filename), in the order bench reports rows. The labels are
# the index column of bench_metrics.csv and the `format` column of parity.csv, so both
# reports go through `load_backends` and stay joinable on backend.
BACKENDS = {
    "torch": ("torch", "model.pt"),
    "tensorrt": ("TensorRT", "model.engine"),
    "openvino": ("OpenVINO", "model.xml"),
    "onnx": ("ONNX", "model.onnx"),
}

WARMUP_ITERS = 10
_synchronize = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)


def _wrapper_class(key: str):
    """Format key -> wrapper class, imported on demand.

    On demand rather than at module scope so a missing tensorrt or openvino install costs
    only its own row instead of the whole benchmark.
    """
    if key == "torch":
        from img_clf.infer.torch_model import TorchModel

        return TorchModel
    if key == "tensorrt":
        from img_clf.infer.trt_model import TRTModel

        return TRTModel
    if key == "openvino":
        from img_clf.infer.ov_model import OVModel

        return OVModel
    if key == "onnx":
        from img_clf.infer.onnx_model import ONNXModel

        return ONNXModel
    raise ValueError(f"no inference wrapper for backend {key!r}")


def load_backends(
    models_dir: Path, requested: Optional[Sequence[str]], half: bool
) -> Iterator[Tuple[str, object]]:
    """Yield (row label, wrapper) for each exported artifact present, in report order."""
    norms = norm_kwargs(models_dir)
    for key in resolve_formats(requested, BACKENDS, "bench.formats"):
        display, filename = BACKENDS[key]
        path = models_dir / filename
        if not path.is_file():
            logger.info(f"Skipping {display}: {filename} not found")
            continue
        try:
            model = _wrapper_class(key)(model_path=str(path), half=half, **norms)
        except Exception as e:
            logger.warning(f"Skipping {display}: {type(e).__name__}: {e}")
            continue
        yield display, model


class CustomDataset(Dataset):
    def __init__(self, root_path: Path, split: pd.DataFrame) -> None:
        self.root_path = root_path
        self.split = split

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        image_path, label = self.split.iloc[idx]
        return image_path, label

    def __len__(self) -> int:
        return len(self.split)


def save_error(
    image: np.ndarray,
    image_path: str,
    gt_label: int,
    pred_label: int,
    prob: float,
    output_path: Path,
    label_to_name: Dict[int, str],
) -> None:
    """Write a misclassified image to `<split>/<gt>_as_<pred>/`.

    Foldering by confusion pair is the point: `beagle_as_english_foxhound` filling up tells
    you which boundary the model cannot see, which a flat dump of failures does not.
    """
    gt_name = label_to_name.get(int(gt_label), str(gt_label))
    pred_name = label_to_name.get(int(pred_label), str(pred_label))
    out_dir = output_path / f"{gt_name}_as_{pred_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    annotated = image.copy()
    vis_one_image(annotated, gt_label, mode="gt", label_to_name=label_to_name)
    vis_one_image(annotated, pred_label, mode="pred", label_to_name=label_to_name, score=prob)
    cv2.imwrite(str(out_dir / f"{Path(image_path).stem}.jpg"), annotated)


def test_model(
    test_loader: DataLoader,
    data_path: Path,
    model,
    name: str,
    errors_path: Optional[Path],
    validator: Validator,
    measure_latency: bool = True,
) -> Tuple[Dict[str, float], List[int], List[int]]:
    """Run `model` over a split. `errors_path` None means "do not dump misclassifications".

    `measure_latency` False skips the warmup and the synchronize pair, for callers that only
    want the predictions (`check_errors`) and should not pay for a benchmark.
    """
    logger.info(f"Testing {name} model")
    label_to_name = validator.label_to_name
    predictions: List[int] = []
    gt_labels: List[int] = []
    latency: List[float] = []

    warmup_done = not measure_latency
    for image_paths, labels in tqdm(test_loader, total=len(test_loader)):
        for im_id, image_path in enumerate(image_paths):
            image = cv2.imread(str(data_path / image_path))
            if image is None:
                logger.warning(f"could not read {image_path}; skipping")
                continue

            if not warmup_done:
                # Lazy allocations, cuDNN autotuning and the first kernel loads all land on
                # the first calls. Averaging them in makes a fast backend look slow.
                for _ in range(WARMUP_ITERS):
                    model(image)
                _synchronize()
                warmup_done = True

            if measure_latency:
                _synchronize()
                t0 = time.perf_counter()
                prediction = model(image)[0]
                _synchronize()
                latency.append((time.perf_counter() - t0) * 1000)
            else:
                prediction = model(image)[0]

            pred_label, max_prob = prediction["label"], prediction["score"]
            gt_label = int(labels[im_id])
            predictions.append(pred_label)
            gt_labels.append(gt_label)
            if errors_path is not None and pred_label != gt_label:
                save_error(
                    image, image_path, gt_label, pred_label, max_prob, errors_path, label_to_name
                )

    metrics, _ = validator.get_metrics(gt_labels, predictions, per_class=False)
    if latency:
        metrics["latency"] = float(np.mean(latency))
    return metrics, gt_labels, predictions


@hydra.main(version_base=None, config_path=config_dir(), config_name=CONFIG_NAME)
def main(cfg: DictConfig):
    data_path = Path(cfg.train.data_path)
    cfg.exp = get_latest_experiment_name(cfg.exp, cfg.train.path_to_save)
    models_dir = Path(cfg.train.path_to_save)
    validator = Validator(len(cfg.train.label_to_name), cfg.train.label_to_name)

    test_csv = data_path / "test.csv"
    val_csv = data_path / "val.csv"
    if not test_csv.is_file() and not val_csv.is_file():
        raise FileNotFoundError(f"no {test_csv} or {val_csv} under {data_path}")
    split_path = test_csv if test_csv.is_file() else val_csv

    test_dataset = CustomDataset(
        root_path=data_path,
        split=pd.read_csv(split_path, header=None),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.train.num_workers,
    )

    errors_root = Path(cfg.train.bench_img_path)
    if errors_root.exists():
        rmtree(errors_root)

    all_metrics = {}
    for name, model in load_backends(models_dir, cfg.bench.formats, cfg.export.half):
        all_metrics[name], _, _ = test_model(
            test_loader,
            data_path,
            model,
            name,
            errors_path=errors_root / name if cfg.train.to_save_errors else None,
            validator=validator,
        )

        # Drop the backend before building the next one: a live TensorRT context or ORT
        # session keeps GPU memory, and the next backend then benchmarks under pressure.
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not all_metrics:
        logger.error(f"No exported artifacts found in {models_dir} - run `make export` first.")
        return

    metrics_df = pd.DataFrame.from_dict(all_metrics, orient="index")
    metrics_df.round(4).to_csv(models_dir / "bench_metrics.csv")
    tabulated_data = tabulate(
        metrics_df.round(4), headers="keys", tablefmt="pretty", showindex=True
    )
    print("\n" + tabulated_data)
    if cfg.train.to_save_errors:
        logger.info(f"Misclassified images saved to: {errors_root}")


if __name__ == "__main__":
    main()
