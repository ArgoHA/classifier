import random
from pathlib import Path
from typing import List, Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from loguru import logger
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from src.dl.utils import seed_worker, vis_one_image


class CustomDataset(Dataset):
    def __init__(
        self,
        img_size: Tuple[int, int],
        root_path: Path,
        split: pd.DataFrame,
        debug_img_processing: bool,
        mode: bool,
        cfg: DictConfig,
    ) -> None:
        self.project_path = Path(cfg.train.root)
        self.root_path = root_path
        self.split = split
        self.target_h, self.target_w = img_size
        self.mode = mode
        self.norm = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.debug_img_processing = debug_img_processing
        self.label_to_name = cfg.train.label_to_name
        self.cases_to_debug = 20
        self._init_augs(cfg)

        self.debug_img_path = Path(cfg.train.debug_img_path)

    def _init_augs(self, cfg) -> None:
        resize = [A.Resize(self.target_h, self.target_w, interpolation=cv2.INTER_AREA)]
        norm = [
            A.Normalize(mean=self.norm[0], std=self.norm[1]),
            ToTensorV2(),
        ]

        if self.mode == "train":
            augs = [
                # A.CoarseDropout(
                #     num_holes_range=(1, 2),
                #     hole_height_range=(0.05, 0.15),
                #     hole_width_range=(0.05, 0.15),
                #     p=cfg.train.augs.coarse_dropout,
                # ),
                A.RandomBrightnessContrast(p=cfg.train.augs.brightness),
                A.RandomGamma(p=cfg.train.augs.gamma),
                A.Blur(p=cfg.train.augs.blur),
                A.GaussNoise(p=cfg.train.augs.noise, std_range=(0.1, 0.2)),
                A.ToGray(p=cfg.train.augs.to_gray),
                A.Affine(
                    rotate=[90, 90],
                    p=cfg.train.augs.rotate_90,
                    fit_output=True,
                ),
                A.HorizontalFlip(p=cfg.train.augs.left_right_flip),
                A.VerticalFlip(p=cfg.train.augs.up_down_flip),
                A.Rotate(
                    limit=cfg.train.augs.rotation_degree,
                    p=cfg.train.augs.rotation_p,
                    interpolation=cv2.INTER_AREA,
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=(114, 114, 114),
                ),
            ]

            self.transform = A.Compose(augs + resize + norm)
        elif self.mode in ["val", "test", "bench"]:
            self.mosaic_prob = 0
            self.transform = A.Compose(resize + norm)
        else:
            raise ValueError(
                f"Unknown mode: {self.mode}, choose from ['train', 'val', 'test', 'bench']"
            )

    def _debug_image(self, image: torch.Tensor, idx: int, label: int, img_path: Path) -> None:
        # Unnormalize the image
        mean = np.array(self.norm[0]).reshape(-1, 1, 1)
        std = np.array(self.norm[1]).reshape(-1, 1, 1)
        image_np = image.cpu().numpy()
        image_np = (image_np * std) + mean

        # Convert from [C, H, W] to [H, W, C]
        image_np = np.transpose(image_np, (1, 2, 0))

        # Convert pixel values from [0, 1] to [0, 255]
        image_np = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)
        image_np = np.ascontiguousarray(image_np)

        # visualize GT label
        vis_one_image(image_np, label, mode="gt", label_to_name=self.label_to_name)

        save_dir = self.debug_img_path / self.mode
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{idx}_idx_{img_path.stem}_debug.jpg"
        cv2.imwrite(str(save_path), cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_path, label = self.split.iloc[idx]

        image = cv2.imread(self.root_path / image_path)
        assert image is not None, f"Image wasn't loaded: {self.root_path / image_path}"

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)["image"]
        label = torch.tensor(label, dtype=torch.long)

        if self.debug_img_processing and idx <= self.cases_to_debug:
            self._debug_image(image, idx, label, Path(image_path))
        return image, label, image_path

    def __len__(self) -> int:
        return len(self.split)


class Loader:
    def __init__(
        self,
        root_path: Path,
        img_size: Tuple[int, int],
        batch_size: int,
        num_workers: int,
        cfg: DictConfig,
        debug_img_processing: bool = False,
    ) -> None:
        self.root_path = root_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cfg = cfg
        self.debug_img_processing = debug_img_processing
        self._get_splits()
        self.class_names = list(cfg.train.label_to_name.values())
        self.label_to_name = cfg.train.label_to_name
        self.multiscale_prob = cfg.train.augs.multiscale_prob
        self.print_class_distribution()

    def _get_splits(self) -> None:
        self.splits = {"train": None, "val": None, "test": None}
        for split_name in self.splits.keys():
            if (self.root_path / f"{split_name}.csv").exists():
                self.splits[split_name] = pd.read_csv(
                    self.root_path / f"{split_name}.csv", header=None
                )
            else:
                self.splits[split_name] = []

    def print_class_distribution(self) -> None:
        all_data = pd.concat([split for split in self.splits.values() if np.any(split)])
        class_counts = all_data[1].value_counts().sort_index()
        class_distribution = {}
        for class_id, count in class_counts.items():
            class_distribution[self.label_to_name[class_id]] = count

    def _build_dataloader_impl(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        collate_fn = self.val_collate_fn
        if dataset.mode == "train":
            collate_fn = self.train_collate_fn

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            collate_fn=collate_fn,
            worker_init_fn=seed_worker,
            prefetch_factor=4,
            pin_memory=True,
        )
        return dataloader

    def build_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        train_ds = CustomDataset(
            self.img_size,
            self.root_path,
            self.splits["train"],
            self.debug_img_processing,
            mode="train",
            cfg=self.cfg,
        )
        val_ds = CustomDataset(
            self.img_size,
            self.root_path,
            self.splits["val"],
            self.debug_img_processing,
            mode="val",
            cfg=self.cfg,
        )

        train_loader = self._build_dataloader_impl(train_ds, shuffle=True)
        val_loader = self._build_dataloader_impl(val_ds)

        test_loader = None
        test_ds = []
        if len(self.splits["test"]):
            test_ds = CustomDataset(
                self.img_size,
                self.root_path,
                self.splits["test"],
                self.debug_img_processing,
                mode="test",
                cfg=self.cfg,
            )
            test_loader = self._build_dataloader_impl(test_ds)

        logger.info(f"Images in train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")
        return train_loader, val_loader, test_loader

    def _collate_fn(self, batch) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        if None in batch:
            return None, None, None
        images, targets, img_paths = zip(*[(item[0], item[1], item[2]) for item in batch])
        return torch.stack(images, dim=0), torch.stack(targets), list(img_paths)

    def val_collate_fn(self, batch) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        return self._collate_fn(batch)

    def train_collate_fn(self, batch) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        During traing add multiscale augmentation to the batch
        """
        images, targets, img_paths = self._collate_fn(batch)

        if random.random() < self.multiscale_prob:
            offset = random.choice([-1, 1]) * 32
            new_h = images.shape[2] + offset
            new_w = images.shape[3] + offset

            images = torch.nn.functional.interpolate(
                images, size=(new_h, new_w), mode="bilinear", align_corners=False
            )
        return images, targets, img_paths
