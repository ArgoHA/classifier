import math
import time
from copy import deepcopy
from pathlib import Path
from shutil import rmtree
from typing import Dict, List, Tuple

import hydra
import numpy as np
import timm
import torch
import wandb
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import autocast, nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dl.dataset import Loader
from src.dl.utils import (
    FocalLoss,
    build_precision_recall_threshold_curves,
    calculate_remaining_time,
    get_vram_usage,
    log_metrics_locally,
    save_metrics,
    set_seeds,
    visualize,
    wandb_logger,
)


def build_model(
    model_name: str, pretrained: bool, num_labels: int, device: str, layers_to_train: int
) -> nn.Module:
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_labels)
    if layers_to_train == -1:
        return model.to(device)
    for param in list(model.parameters())[:-layers_to_train]:
        param.requires_grad = False
    return model.to(device)


def prepare_model(model_name: str, model_path: Path, num_labels: int, device: str) -> nn.Module:
    model = build_model(
        model_name=model_name,
        num_labels=num_labels,
        pretrained=False,
        device=device,
        layers_to_train=-1,
    )
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"), weights_only=True)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


class ModelEMA:
    def __init__(self, student, ema_momentum):
        self.model = deepcopy(student).eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.ema_scheduler = lambda x: ema_momentum * (1 - math.exp(-x / 2000))

    def update(self, iters, student):
        student = student.state_dict()
        with torch.no_grad():
            momentum = self.ema_scheduler(iters)
            for name, param in self.model.state_dict().items():
                if param.dtype.is_floating_point:
                    param *= momentum
                    param += (1.0 - momentum) * student[name].detach()


class Trainer:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.device = cfg.train.device
        self.epochs = cfg.train.epochs
        self.path_to_save = Path(cfg.train.path_to_save)
        self.to_visualize_eval = cfg.train.to_visualize_eval
        self.amp_enabled = cfg.train.amp_enabled
        self.clip_max_norm = cfg.train.clip_max_norm
        self.b_accum_steps = max(cfg.train.b_accum_steps, 1)
        self.early_stopping = cfg.train.early_stopping
        self.use_wandb = cfg.train.use_wandb
        self.label_to_name = cfg.train.label_to_name
        self.n_labels = len(self.label_to_name)

        self.debug_img_path = Path(self.cfg.train.debug_img_path)
        self.eval_preds_path = Path(self.cfg.train.eval_preds_path)
        self.init_dirs()

        if self.use_wandb:
            wandb.init(
                project=cfg.project_name,
                name=cfg.exp,
                config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            )

        log_file = Path(cfg.train.path_to_save) / "train_log.txt"
        log_file.unlink(missing_ok=True)
        logger.add(log_file, format="{message}", level="INFO", rotation="10 MB")

        set_seeds(cfg.train.seed, cfg.train.cudnn_fixed)

        base_loader = Loader(
            root_path=Path(cfg.train.data_path),
            img_size=tuple(cfg.train.img_size),
            batch_size=cfg.train.batch_size,
            num_workers=cfg.train.num_workers,
            cfg=cfg,
            debug_img_processing=cfg.train.debug_img_processing,
        )
        self.train_loader, self.val_loader, self.test_loader = base_loader.build_dataloaders()

        self.model = build_model(
            model_name=cfg.model_name,
            num_labels=self.n_labels,
            pretrained=cfg.train.pretrained,
            device=self.device,
            layers_to_train=cfg.train.layers_to_train,
        )

        self.ema_model = None
        if self.cfg.train.use_ema:
            logger.info("EMA model will be evaluated and saved")
            self.ema_model = ModelEMA(self.model, cfg.train.ema_momentum)

        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.train.label_smoothing)
        # self.loss_fn = FocalLoss(
        #     gamma=2.0, alpha=None, label_smoothing=cfg.train.label_smoothing, reduction="mean"
        # )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=cfg.train.base_lr, weight_decay=cfg.train.weight_decay
        )

        # class_weights = None
        # if cfg.train.class_weights:
        #     class_weights = torch.tensor(
        #         cfg.train.class_weights, dtype=torch.float32, device=self.device
        #     )
        # self.loss_fn = nn.CrossEntropyLoss(
        #     label_smoothing=cfg.train.label_smoothing, weight=class_weights
        # )

        # self.optimizer = torch.optim.Adam(
        #     self.model.parameters(),
        #     lr=cfg.train.base_lr,
        #     weight_decay=cfg.train.weight_decay,
        #     # betas=cfg.train.betas,
        # )

        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=cfg.train.base_lr * 10,
            epochs=cfg.train.epochs,
            steps_per_epoch=len(self.train_loader) // self.b_accum_steps,
            pct_start=cfg.train.cycler_pct_start,
            cycle_momentum=False,
        )

        if self.amp_enabled:
            self.scaler = GradScaler()

        if self.use_wandb:
            wandb.watch(self.model)

    def init_dirs(self):
        for path in [self.debug_img_path, self.eval_preds_path]:
            if path.exists():
                rmtree(path)
            path.mkdir(exist_ok=True, parents=True)

        self.path_to_save.mkdir(exist_ok=True, parents=True)
        with open(self.path_to_save / "config.yaml", "w") as f:
            OmegaConf.save(config=self.cfg, f=f)

    @staticmethod
    def get_metrics(
        gt_labels: List[int], preds: List[int], per_class: bool, label_to_name=None
    ) -> Dict[str, float]:
        num_labels = len(set(gt_labels))
        if num_labels == 2:
            average = "binary"
        else:
            average = "macro"

        metrics = {}
        metrics["accuracy"] = accuracy_score(gt_labels, preds)
        metrics["f1"] = f1_score(gt_labels, preds, average=average)
        metrics["precision"] = precision_score(gt_labels, preds, average=average)
        metrics["recall"] = recall_score(gt_labels, preds, average=average)

        if not per_class or num_labels <= 2:
            return metrics, None

        per_class_metrics = {}
        unique_labels = sorted(set(gt_labels))
        f1s = f1_score(gt_labels, preds, average=None, labels=unique_labels)
        precisions = precision_score(gt_labels, preds, average=None, labels=unique_labels)
        recalls = recall_score(gt_labels, preds, average=None, labels=unique_labels)
        accs = []

        for cl in unique_labels:
            idx = [i for i, lbl in enumerate(gt_labels) if lbl == cl]
            acc = sum(1 for i in idx if preds[i] == cl) / len(idx) if idx else 0.0
            accs.append(acc)

        for i, cl in enumerate(unique_labels):
            class_name = label_to_name.get(cl, cl) if label_to_name else cl
            per_class_metrics[class_name] = {
                "accuracy": accs[i],
                "f1": f1s[i],
                "precision": precisions[i],
                "recall": recalls[i],
            }
        return metrics, per_class_metrics

    def postprocess(
        self, probs: torch.Tensor, gt_labels: torch.Tensor
    ) -> Tuple[List[int], List[int]]:
        preds = torch.argmax(probs, dim=1).tolist()
        gt_labels = gt_labels.tolist()
        return preds, gt_labels

    def evaluate(
        self,
        test_loader: DataLoader,
        model: nn.Module,
        device: str,
        path_to_save: Path,
        mode: str,
        per_class: bool,
    ) -> Dict[str, float]:
        probs, gt_labels = self.get_full_preds(model, test_loader, device)

        if path_to_save is not None:
            for class_idx in range(self.n_labels):
                output_path = path_to_save / "pr_curves"
                output_path.mkdir(exist_ok=True)

                build_precision_recall_threshold_curves(
                    gt_labels,
                    probs[:, class_idx],
                    output_path / f"{mode}_pr_curve_class_{self.label_to_name[class_idx]}.png",
                    class_idx,
                )

        preds, gt_labels = self.postprocess(probs, gt_labels)
        metrics, per_class_metrics = self.get_metrics(
            gt_labels, preds, per_class, self.label_to_name
        )
        return metrics, per_class_metrics

    def get_full_preds(
        self, model: nn.Module, val_loader: DataLoader, device: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        val_probs = []  # List to store predicted probabilities for all classes
        val_labels = []
        model.eval()

        with torch.no_grad():
            for idx, (inputs, labels, img_paths) in enumerate(val_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                logits = model.forward(inputs)
                probs = torch.softmax(logits, dim=1)

                val_probs.append(probs)
                val_labels.extend(labels)

                if self.to_visualize_eval and idx <= 1:
                    visualize(
                        img_paths,
                        labels,
                        probs,
                        dataset_path=Path(self.cfg.train.data_path),
                        path_to_save=self.eval_preds_path,
                        label_to_name=self.label_to_name,
                    )

        val_probs = torch.cat(val_probs, dim=0)
        val_labels = torch.tensor(val_labels)
        return val_probs, val_labels

    def save_model(self, metrics, best_metric):
        model_to_save = self.model
        if self.ema_model:
            model_to_save = self.ema_model.model

        self.path_to_save.mkdir(parents=True, exist_ok=True)
        torch.save(model_to_save.state_dict(), self.path_to_save / "last.pt")

        decision_metric = metrics["f1"]
        if decision_metric > best_metric:
            best_metric = decision_metric
            logger.info("Saving new best model🔥")
            torch.save(model_to_save.state_dict(), self.path_to_save / "model.pt")
            self.early_stopping_steps = 0
        else:
            self.early_stopping_steps += 1
        return best_metric

    def train(self) -> None:
        best_metric = 0
        cur_iter = 0
        ema_iter = 0
        self.early_stopping_steps = 0
        one_epoch_time = None

        def optimizer_step(step_scheduler: bool):
            """
            Clip grads, optimizer step, scheduler step, zero grad, EMA model update
            """
            nonlocal ema_iter
            if self.amp_enabled:
                if self.clip_max_norm:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_max_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()

            else:
                if self.clip_max_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_max_norm)
                self.optimizer.step()

            if step_scheduler:
                self.scheduler.step()
            self.optimizer.zero_grad()

            if self.ema_model:
                ema_iter += 1
                self.ema_model.update(ema_iter, self.model)

        for epoch in range(1, self.epochs + 1):
            epoch_start_time = time.time()
            self.model.train()
            losses = []

            with tqdm(self.train_loader, unit="batch") as tepoch:
                for batch_idx, (inputs, labels, _) in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {epoch}/{self.epochs}")
                    if inputs is None:
                        continue
                    cur_iter += 1

                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    lr = self.optimizer.param_groups[0]["lr"]

                    if self.amp_enabled:
                        with autocast(device_type=self.device, dtype=torch.float16):
                            output = self.model(inputs)
                            loss = self.loss_fn(output, labels)
                        self.scaler.scale(loss).backward()

                    else:
                        output = self.model(inputs)
                        loss = self.loss_fn(output, labels)
                        loss.backward()

                    if (batch_idx + 1) % self.b_accum_steps == 0:
                        optimizer_step(step_scheduler=True)

                    losses.append(loss.item())

                    tepoch.set_postfix(
                        loss=np.mean(losses) * self.b_accum_steps,
                        eta=calculate_remaining_time(
                            one_epoch_time,
                            epoch_start_time,
                            epoch,
                            self.epochs,
                            cur_iter,
                            len(self.train_loader),
                        ),
                        vram=f"{get_vram_usage()}%",
                    )

            # Final update for any leftover gradients from an incomplete accumulation step
            if (batch_idx + 1) % self.b_accum_steps != 0:
                optimizer_step(step_scheduler=False)

            if self.use_wandb:
                wandb.log({"lr": lr, "epoch": epoch})

            metrics, _ = self.evaluate(
                test_loader=self.val_loader,
                model=self.model,
                device=self.device,
                path_to_save=None,
                mode="val",
                per_class=False,
            )

            best_metric = self.save_model(metrics, best_metric)
            save_metrics(
                {},
                metrics,
                np.mean(losses) * self.b_accum_steps,
                epoch,
                path_to_save=None,
                use_wandb=self.use_wandb,
            )

            one_epoch_time = time.time() - epoch_start_time

            if self.early_stopping and self.early_stopping_steps >= self.early_stopping:
                logger.info("Early stopping")
                break


@hydra.main(version_base=None, config_path="../../", config_name="config")
def main(cfg: DictConfig) -> None:
    trainer = Trainer(cfg)

    try:
        t_start = time.time()
        trainer.train()
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
    except Exception as e:
        logger.error(e)
    finally:
        logger.info("Evaluating best model...")
        model = prepare_model(
            model_name=cfg.model_name,
            model_path=Path(cfg.train.path_to_save) / "model.pt",
            num_labels=len(cfg.train.label_to_name),
            device=cfg.train.device,
        )
        if trainer.ema_model:
            trainer.ema_model.model = model
        else:
            trainer.model = model

        val_metrics, val_per_class_metrics = trainer.evaluate(
            test_loader=trainer.val_loader,
            model=model,
            device=cfg.train.device,
            path_to_save=Path(cfg.train.path_to_save),
            mode="val",
            per_class=True,
        )
        if cfg.train.use_wandb:
            wandb_logger(None, val_metrics, epoch=cfg.train.epochs + 1, mode="val")

        test_metrics = {}
        test_per_class_metrics = {}
        if trainer.test_loader:
            test_metrics, test_per_class_metrics = trainer.evaluate(
                test_loader=trainer.test_loader,
                model=model,
                device=cfg.train.device,
                path_to_save=Path(cfg.train.path_to_save),
                mode="test",
                per_class=True,
            )
            if cfg.train.use_wandb:
                wandb_logger(None, test_metrics, epoch=-1, mode="test")

        log_metrics_locally(
            all_metrics={"val": val_metrics, "test": test_metrics},
            path_to_save=Path(cfg.train.path_to_save),
            epoch=0,
            per_class={
                "val": val_per_class_metrics,
                "test": test_per_class_metrics,
            },
        )
        logger.info(f"Full training time: {(time.time() - t_start) / 60 / 60:.2f} hours")


if __name__ == "__main__":
    main()
