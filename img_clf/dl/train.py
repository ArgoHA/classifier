import math
import time
from copy import deepcopy
from pathlib import Path
from shutil import rmtree
from typing import Dict, Optional, Sequence, Tuple

import hydra
import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from timm.data.mixup import mixup_target
from timm.loss import SoftTargetCrossEntropy
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from img_clf.config.resolve import CONFIG_NAME, config_dir
from img_clf.dl.ckpt import ckpt_meta, resolve_norm, save_checkpoint
from img_clf.dl.dataset import Loader
from img_clf.dl.model import build_model, head_param_names, prepare_model
from img_clf.dl.utils import (
    auto_batch_size,
    build_precision_recall_threshold_curves,
    calculate_remaining_time,
    get_vram_usage,
    log_metrics_locally,
    save_metrics,
    set_seeds,
    visualize,
    wandb_logger,
)
from img_clf.dl.validator import Validator


def build_optimizer(
    model: nn.Module,
    base_lr: float,
    weight_decay: float,
    betas: Sequence[float],
    backbone_lr: Optional[float] = None,
) -> torch.optim.Optimizer:
    """AdamW with weight decay switched off for norm and bias parameters.
    Decaying a norm's scale pulls it toward zero, which is a change of function, not
    regularization; the same goes for biases.
    """
    backbone_decay, backbone_no_decay, head_decay, head_no_decay = [], [], [], []
    head_names = head_param_names(model)
    if not head_names:
        logger.warning(
            f"{type(model).__name__} exposes no classifier parameters; every tensor will "
            "train at the backbone learning rate"
        )
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # 1-D tensors are norms' weight/bias and every bias; those are the no-decay set.
        no_decay = param.ndim <= 1 or name.endswith(".bias")
        if name in head_names:
            (head_no_decay if no_decay else head_decay).append(param)
        else:
            (backbone_no_decay if no_decay else backbone_decay).append(param)

    # `is None`, not falsy: backbone_lr=0.0 is the standard way to freeze the backbone
    # while the head still trains, and would otherwise become base_lr.
    bb_lr = base_lr if backbone_lr is None else backbone_lr
    groups = [
        {"params": backbone_decay, "lr": bb_lr, "initial_lr": bb_lr, "weight_decay": weight_decay},
        {"params": backbone_no_decay, "lr": bb_lr, "initial_lr": bb_lr, "weight_decay": 0.0},
        {"params": head_decay, "lr": base_lr, "initial_lr": base_lr, "weight_decay": weight_decay},
        {"params": head_no_decay, "lr": base_lr, "initial_lr": base_lr, "weight_decay": 0.0},
    ]
    counts = [len(g["params"]) for g in groups]
    logger.info(
        f"AdamW groups (backbone/head x decay/no-decay): {counts}, "
        f"backbone_lr={bb_lr:g}, head_lr={base_lr:g}"
    )
    return torch.optim.AdamW(groups, lr=base_lr, betas=tuple(betas), weight_decay=weight_decay)


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
        # bfloat16 by default
        self.amp_dtype = (
            torch.float16 if cfg.train.get("amp_dtype") == "float16" else torch.bfloat16
        )
        if self.amp_enabled and self.amp_dtype is torch.bfloat16 and self.device == "cuda":
            assert torch.cuda.is_bf16_supported(), (
                "bf16 AMP is not supported on this GPU; set train.amp_dtype=float16"
            )
        self.decision_metrics = list(cfg.train.get("decision_metrics", ["f1"]))
        self.clip_max_norm = cfg.train.clip_max_norm
        self.b_accum_steps = max(cfg.train.b_accum_steps, 1)
        self.early_stopping = cfg.train.early_stopping
        self.use_wandb = cfg.train.use_wandb
        self.label_to_name = cfg.train.label_to_name
        self.n_labels = len(self.label_to_name)
        # Averaging mode comes from this configured count, never from the labels present
        # in a given split (see validator.py).
        self.validator = Validator(self.n_labels, self.label_to_name)

        self.debug_img_path = Path(self.cfg.train.debug_img_path)
        self.eval_preds_path = Path(self.cfg.train.eval_preds_path)
        self.init_dirs()

        if self.use_wandb:
            try:
                wandb.init(
                    project=cfg.project_name,
                    name=cfg.exp,
                    config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
                    settings=wandb.Settings(init_timeout=30),
                )
            except Exception as e:
                logger.warning(f"wandb.init failed ({type(e).__name__}: {e}); continuing without")
                self.use_wandb = False

        log_file = Path(cfg.train.path_to_save) / "train_log.txt"
        log_file.unlink(missing_ok=True)
        logger.add(log_file, format="{message}", level="INFO", rotation="10 MB")

        set_seeds(cfg.train.seed, cfg.train.cudnn_fixed)

        self.norm = resolve_norm(cfg.model_name)
        logger.info(f"Normalization for {cfg.model_name}: mean={self.norm[0]} std={self.norm[1]}")

        batch_size = int(cfg.train.batch_size)
        if batch_size == -1:
            batch_size = auto_batch_size(cfg, self.n_labels, self.device)

        base_loader = Loader(
            root_path=Path(cfg.train.data_path),
            img_size=tuple(cfg.train.img_size),
            batch_size=batch_size,
            num_workers=cfg.train.num_workers,
            cfg=cfg,
            debug_img_processing=cfg.train.debug_img_processing,
            norm=self.norm,
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

        self.mixup_fn = base_loader.mixup_fn
        if self.mixup_fn is not None:
            # timm's Mixup emits soft targets and applies label smoothing itself, so the
            # hard-label CrossEntropy (and its own smoothing) would double-count it.
            self.loss_fn = SoftTargetCrossEntropy()
        else:
            self.loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.train.label_smoothing)

        self.optimizer = build_optimizer(
            self.model,
            base_lr=cfg.train.base_lr,
            weight_decay=cfg.train.weight_decay,
            betas=cfg.train.betas,
            backbone_lr=cfg.train.get("backbone_lr", None),
        )

        max_lr_mult = float(cfg.train.get("max_lr_mult", 10))
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=[g["initial_lr"] * max_lr_mult for g in self.optimizer.param_groups],
            epochs=cfg.train.epochs,
            steps_per_epoch=len(self.train_loader) // self.b_accum_steps,
            pct_start=cfg.train.cycler_pct_start,
            cycle_momentum=False,
        )

        if self.amp_enabled:
            # Disabled for bf16, which makes scale/unscale/step pass straight through.
            self.scaler = GradScaler(enabled=self.amp_dtype is torch.float16)

    def init_dirs(self):
        for path in [self.debug_img_path, self.eval_preds_path]:
            if path.exists():
                rmtree(path)
            path.mkdir(exist_ok=True, parents=True)

        self.path_to_save.mkdir(exist_ok=True, parents=True)
        # Resolved, not raw: this copy is the standalone record of the run, and
        # ckpt.sibling_config reads it to describe a checkpoint whose meta is incomplete.
        # Left unresolved, every path in it is a literal "${train.root}/..." string.
        with open(self.path_to_save / "config.yaml", "w") as f:
            OmegaConf.save(config=OmegaConf.to_container(self.cfg, resolve=True), f=f)

    def evaluate(
        self,
        test_loader: DataLoader,
        model: nn.Module,
        path_to_save: Path,
        mode: str,
        per_class: bool,
    ) -> Tuple[Dict[str, float], Optional[Dict[str, Dict[str, float]]], Optional[Dict[str, float]]]:
        """-> (metrics, per-class metrics, F1-optimal threshold).

        The threshold used to be handed back through `self.best_threshold`, which made two
        consecutive calls order-dependent and invisible in the signature.
        """
        probs, gt_labels = self.get_full_preds(model, test_loader)

        if path_to_save is not None:
            for class_idx in range(self.n_labels):
                output_path = path_to_save / "pr_curves"
                output_path.mkdir(exist_ok=True, parents=True)

                build_precision_recall_threshold_curves(
                    gt_labels,
                    probs[:, class_idx],
                    output_path / f"{mode}_pr_curve_class_{self.label_to_name[class_idx]}.png",
                    class_idx,
                )

        preds, gt_int = self.validator.postprocess(probs, gt_labels)
        metrics, per_class_metrics = self.validator.get_metrics(gt_int, preds, per_class)

        best_threshold = None
        if path_to_save is not None:
            self.validator.save_confusion_matrix(gt_int, preds, path_to_save, mode)
            # Only a binary task has a threshold to sweep
            if self.n_labels == 2:
                best_threshold = self.validator.threshold_sweep(
                    probs.cpu().numpy(), gt_int, path_to_save, mode
                )
        return metrics, per_class_metrics, best_threshold

    def get_full_preds(
        self, model: nn.Module, val_loader: DataLoader
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        val_probs = []  # List to store predicted probabilities for all classes
        val_labels = []
        model.eval()

        with torch.no_grad():
            for idx, (inputs, labels, img_paths) in enumerate(val_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
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
        meta = ckpt_meta(self.cfg, *self.norm)
        save_checkpoint(self.path_to_save / "last.pt", model_to_save.state_dict(), meta)

        available = [m for m in self.decision_metrics if m in metrics]
        if not available:
            raise KeyError(
                f"none of train.decision_metrics {self.decision_metrics} is reported; "
                f"available: {sorted(metrics)}"
            )
        decision_metric = float(np.mean([metrics[m] for m in available]))
        if decision_metric > best_metric:
            best_metric = decision_metric
            logger.info("Saving new best model🔥")
            save_checkpoint(self.path_to_save / "model.pt", model_to_save.state_dict(), meta)
            self.early_stopping_steps = 0
        else:
            self.early_stopping_steps += 1
        return best_metric

    def _apply_mixup(self, inputs, labels):
        """Mix the batch, or produce matching soft targets when the batch is odd.

        timm's Mixup mirrors the batch - element i mixes with element n-1-i - so it asserts
        an even length, and the last batch of an epoch usually is not (59 images here, of
        10,363). Dropping that batch would discard real images every epoch, so instead it
        skips the mixing and gets the smoothed one-hot targets mixup itself would emit at
        lam=1. SoftTargetCrossEntropy then sees the same shape for every batch, mixed or not.
        """
        if self.mixup_fn is None:
            return inputs, labels
        if inputs.shape[0] % 2 == 0:
            return self.mixup_fn(inputs, labels)
        return inputs, mixup_target(
            labels, self.n_labels, lam=1.0, smoothing=self.cfg.train.label_smoothing
        )

    def _set_train_mode(self) -> None:
        """`model.train()`, then put any frozen groups back into eval.

        Freezing only clears requires_grad; a BatchNorm in train mode still updates its
        running statistics from every batch, so a "frozen" backbone would keep drifting.
        """
        self.model.train()
        for name in getattr(self.model, "_frozen_groups", []):
            getattr(self.model, name).eval()

    def evaluate_best(self, t_start: float) -> None:
        """Reload the best checkpoint, score val/test, write the run's report."""
        cfg = self.cfg
        logger.info("Evaluating best model...")
        model = prepare_model(
            model_name=cfg.model_name,
            model_path=self.path_to_save / "model.pt",
            num_labels=self.n_labels,
            device=self.device,
        )
        thresholds = {}
        val_metrics, val_per_class_metrics, thresholds["val"] = self.evaluate(
            test_loader=self.val_loader,
            model=model,
            path_to_save=self.path_to_save,
            mode="val",
            per_class=True,
        )
        # self.use_wandb, not the config: it is cleared when wandb.init fails, and the
        # config still says True.
        if self.use_wandb:
            wandb_logger(None, val_metrics, epoch=cfg.train.epochs + 1, mode="val")

        test_metrics = {}
        test_per_class_metrics = {}
        if self.test_loader:
            test_metrics, test_per_class_metrics, thresholds["test"] = self.evaluate(
                test_loader=self.test_loader,
                model=model,
                path_to_save=self.path_to_save,
                mode="test",
                per_class=True,
            )
            if self.use_wandb:
                wandb_logger(None, test_metrics, epoch=-1, mode="test")

        log_metrics_locally(
            all_metrics={"val": val_metrics, "test": test_metrics},
            path_to_save=self.path_to_save,
            epoch=0,
        )
        self.validator.save_extended_metrics(
            self.path_to_save,
            {"val": val_per_class_metrics, "test": test_per_class_metrics},
            thresholds,
        )
        logger.info(f"Full training time: {(time.time() - t_start) / 60 / 60:.2f} hours")

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
            self._set_train_mode()
            losses = []

            with tqdm(self.train_loader, unit="batch") as tepoch:
                for batch_idx, (inputs, labels, _) in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {epoch}/{self.epochs}")
                    if inputs is None:
                        continue
                    cur_iter += 1

                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    # Train only: validation always scores against hard labels.
                    inputs, labels = self._apply_mixup(inputs, labels)

                    lr = self.optimizer.param_groups[0]["lr"]

                    if self.amp_enabled:
                        with autocast(device_type=self.device, dtype=self.amp_dtype):
                            output = self.model(inputs)
                        # Cross-entropy in fp32: the log-softmax is the one place in this
                        # graph where reduced precision actually costs accuracy.
                        with autocast(device_type=self.device, enabled=False):
                            loss = self.loss_fn(output.float(), labels)
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

            epoch_time = time.time() - epoch_start_time
            mean_loss = float(np.mean(losses)) * self.b_accum_steps
            logger.info(
                f"Epoch {epoch}/{self.epochs} | loss {mean_loss:.4f} | lr {lr:.2e} | "
                f"{epoch_time:.1f}s | vram {get_vram_usage()}%"
            )

            metrics, _, _ = self.evaluate(
                test_loader=self.val_loader,
                model=self.model,
                path_to_save=None,
                mode="val",
                per_class=False,
            )

            best_metric = self.save_model(metrics, best_metric)
            save_metrics(
                {},
                metrics,
                mean_loss,
                epoch,
                path_to_save=None,
                use_wandb=self.use_wandb,
            )

            one_epoch_time = epoch_time

            if self.early_stopping and self.early_stopping_steps >= self.early_stopping:
                logger.info("Early stopping")
                break


@hydra.main(version_base=None, config_path=config_dir(), config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    trainer = Trainer(cfg)

    fatal_error = None
    try:
        t_start = time.time()
        trainer.train()
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
    except Exception as e:
        # A mid-training OOM (or any hard failure) must not be swallowed
        logger.exception(e)
        fatal_error = e
    finally:
        ckpt_path = Path(cfg.train.path_to_save) / "model.pt"
        if fatal_error is not None:
            logger.error("Training failed; skipping best-model evaluation")
        elif not ckpt_path.is_file():
            # Interrupted before the first epoch finished, so nothing was ever saved.
            logger.error(f"No checkpoint at {ckpt_path}; skipping best-model evaluation")
        else:
            trainer.evaluate_best(t_start)

    if fatal_error is not None:
        raise fatal_error


if __name__ == "__main__":
    main()
