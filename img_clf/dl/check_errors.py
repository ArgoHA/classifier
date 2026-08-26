"""Dump misclassified images for a split, foldered by confusion pair.

`make bench` already does this for every exported backend while it measures latency; this is
the standalone version for when you only want to look at the mistakes - it runs one backend
over one split and writes nothing else. The walk itself is `bench.test_model` with the
timing turned off, so the two cannot disagree about how a split is scored.

Output: `output/check_errors/<split>/<gt>_as_<pred>/<image>.jpg`, so the directory listing
ranks the model's confusions by size.
"""

from pathlib import Path
from shutil import rmtree

import hydra
import pandas as pd
from loguru import logger
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from img_clf.config.resolve import CONFIG_NAME, config_dir
from img_clf.dl.bench import CustomDataset, test_model
from img_clf.dl.utils import get_latest_experiment_name
from img_clf.dl.validator import Validator
from img_clf.dl.ckpt import norm_kwargs
from img_clf.infer.torch_model import TorchModel


@hydra.main(version_base=None, config_path=config_dir(), config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    cfg.exp = get_latest_experiment_name(cfg.exp, cfg.train.path_to_save)
    data_path = Path(cfg.train.data_path)
    models_dir = Path(cfg.train.path_to_save)

    split_name = str(cfg.check_errors.split)
    csv_path = data_path / f"{split_name}.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"{csv_path} not found - run `make split` first.")

    model_path = models_dir / "model.pt"
    if not model_path.is_file():
        raise FileNotFoundError(f"{model_path} not found - train first.")
    # norm_kwargs, same as bench: scoring the same split on different preprocessing would
    # make the confusion-pair dump disagree with bench_metrics.csv for the same weights.
    model = TorchModel(model_path=str(model_path), half=cfg.export.half, **norm_kwargs(models_dir))

    output_path = Path(cfg.train.root) / "output" / "check_errors" / split_name
    if output_path.exists():
        rmtree(output_path)

    split = pd.read_csv(csv_path, header=None)
    loader = DataLoader(
        CustomDataset(root_path=data_path, split=split),
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
    )
    validator = Validator(len(cfg.train.label_to_name), cfg.train.label_to_name)

    metrics, gt_labels, predictions = test_model(
        loader,
        data_path,
        model,
        split_name,
        errors_path=output_path,
        validator=validator,
        measure_latency=False,
    )

    n_errors = sum(g != p for g, p in zip(gt_labels, predictions))
    logger.info(
        f"{split_name}: {n_errors}/{len(gt_labels)} wrong "
        f"(accuracy {metrics['accuracy']:.4f}, f1 {metrics['f1']:.4f})"
    )

    if output_path.exists():
        pairs = sorted(
            ((len(list(d.glob("*.jpg"))), d.name) for d in output_path.iterdir() if d.is_dir()),
            reverse=True,
        )
        for count, name in pairs[:10]:
            logger.info(f"  {count:4d}  {name}")
    logger.info(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
