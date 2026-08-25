from pathlib import Path
from shutil import copyfile, rmtree
from typing import Dict, List

import cv2
import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from src.dl.utils import get_latest_experiment_name
from src.infer.torch_model import Torch_model


def run_prod_infer(
    model: Torch_model, path_to_data: Path, output_path: Path, label_to_name: Dict[int, str]
) -> List[int]:
    preds = []
    img_paths = [
        x for x in Path(path_to_data).glob("*") if x.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]

    for img_path in tqdm(img_paths):
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = model(img)

        f_res = res[0]
        preds.append(f_res)
        save_pred(img_path, label_to_name[f_res], output_path)
    return preds


def save_pred(img_path, class_name, output_path):
    output_path = output_path / class_name
    output_path.mkdir(exist_ok=True, parents=True)
    copyfile(img_path, output_path / img_path.name)


@hydra.main(version_base=None, config_path="../../", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg.exp = get_latest_experiment_name(cfg.exp, cfg.train.path_to_save)
    model = Torch_model(
        model_name=cfg.model_name,
        model_path=str(Path(cfg.train.path_to_save) / "model.pt"),
        n_outputs=len(cfg.train.label_to_name),
        input_size=cfg.train.img_size,
    )

    output_path = Path(cfg.train.infer_path)
    if output_path.exists():
        rmtree(output_path)

    run_prod_infer(
        model=model,
        path_to_data=Path(cfg.train.path_to_test_data),
        output_path=output_path,
        label_to_name=cfg.train.label_to_name,
    )


if __name__ == "__main__":
    main()
