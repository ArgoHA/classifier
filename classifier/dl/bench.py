import time
from pathlib import Path
from shutil import rmtree
from typing import Tuple

import cv2
import hydra
import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import DictConfig
from tabulate import tabulate
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.dl.train import Trainer
from src.dl.utils import get_latest_experiment_name
from src.infer.onnx_model import ONNX_model
from src.infer.ov_model import OV_model
from src.infer.torch_model import Torch_model
from src.infer.trt_model import TensorRT_model


class CustomDataset(Dataset):
    def __init__(self, root_path: Path, split: pd.DataFrame) -> None:
        self.root_path = root_path
        self.split = split

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        image_path, label = self.split.iloc[idx]
        return image_path, label

    def __len__(self) -> int:
        return len(self.split)


def save_errors():
    pass


def test_model(test_loader: DataLoader, data_path: Path, model, name: str, to_save_errors: bool):
    logger.info(f"Testing {name} model")
    predictions = []
    gt_labels = []
    latency = []

    for batch in tqdm(test_loader, total=len(test_loader)):
        image_paths, labels = batch
        batch_predictions = []
        for im_id, image_path in enumerate(image_paths):
            image = cv2.imread(data_path / image_path)

            t0 = time.perf_counter()
            pred_label, max_prob = model(image)
            latency.append((time.perf_counter() - t0) * 1000)
            batch_predictions.append(pred_label)

            if to_save_errors and pred_label != labels[im_id]:
                save_errors()

        predictions.extend(batch_predictions)
        gt_labels.extend(labels.tolist())

    metrics, _ = Trainer.get_metrics(gt_labels, predictions, per_class=False)
    metrics["latency"] = np.mean(latency[1:])
    return metrics


@hydra.main(version_base=None, config_path="../../", config_name="config")
def main(cfg: DictConfig):
    data_path = Path(cfg.train.data_path)
    cfg.exp = get_latest_experiment_name(cfg.exp, cfg.train.path_to_save)

    torch_model = Torch_model(
        model_name=cfg.model_name,
        model_path=str(Path(cfg.train.path_to_save) / "model.pt"),
        n_outputs=len(cfg.train.label_to_name),
        input_size=cfg.train.img_size,
        half=cfg.export.half,
    )

    trt_model = TensorRT_model(
        model_path=str(Path(cfg.train.path_to_save) / "model.engine"),
        n_outputs=len(cfg.train.label_to_name),
        input_size=cfg.train.img_size,
        half=cfg.export.half,
    )
    ov_model = OV_model(
        model_path=str(Path(cfg.train.path_to_save) / "model.xml"),
        n_outputs=len(cfg.train.label_to_name),
        input_size=cfg.train.img_size,
        half=cfg.export.half,
    )

    onnx_model = ONNX_model(
        model_path=str(Path(cfg.train.path_to_save) / "model.onnx"),
        n_outputs=len(cfg.train.label_to_name),
        input_size=cfg.train.img_size,
        half=cfg.export.half,
    )

    test_dataset = CustomDataset(
        root_path=data_path,
        split=pd.read_csv(data_path / "test.csv", header=None),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
    )

    output_path = Path(cfg.train.bench_img_path)
    if output_path.exists():
        rmtree(output_path)

    all_metrics = {}
    models = {
        "torch": torch_model,
        "TensorRT": trt_model,
        "OV": ov_model,
        "ONNX": onnx_model,
    }
    for model_name, model in models.items():
        all_metrics[model_name] = test_model(
            test_loader, data_path, model, model_name, to_save_errors=cfg.train.to_save_errors
        )

    metrics_df = pd.DataFrame.from_dict(all_metrics, orient="index")
    tabulated_data = tabulate(
        metrics_df.round(4), headers="keys", tablefmt="pretty", showindex=True
    )
    print("\n" + tabulated_data)


if __name__ == "__main__":
    main()
