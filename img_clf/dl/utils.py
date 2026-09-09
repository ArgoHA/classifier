import random
import subprocess
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from loguru import logger
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from sklearn.metrics import precision_recall_curve
from tabulate import tabulate

import wandb
from img_clf.dl.model import build_model


def build_precision_recall_threshold_curves(
    gt_labels: torch.Tensor, probs: torch.Tensor, output_path: Path, class_idx: int
) -> None:
    gt_labels = gt_labels.cpu().numpy()
    probs = probs.cpu().numpy()

    # Convert the multi-class labels to binary labels for the current class
    binary_gt_labels = (gt_labels == class_idx).astype(int)

    precision, recall, thresholds = precision_recall_curve(binary_gt_labels, probs)
    thresholds = np.append(thresholds, 1)  # appending 1 to the end to match the length

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precision, label="Precision", color="blue")
    plt.plot(thresholds, recall, label="Recall", color="red")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(f"Precision & Recall vs. Threshold Curve for Class {class_idx}")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()


def set_seeds(seed: int, cudnn_fixed: bool = False) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if cudnn_fixed:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):  # noqa
    """Seed every source of randomness a dataloader worker owns.

    torch.initial_seed() is already per-worker and derived from the base seed, so this is
    reproducible run to run. albumentations needs its own call: its generator is
    independent of numpy/random, so without this every worker augments from an
    entropy-seeded stream and the whole run becomes unreproducible.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

    info = torch.utils.data.get_worker_info()
    transform = getattr(getattr(info, "dataset", None), "transform", None)
    if transform is not None and hasattr(transform, "set_random_seed"):
        transform.set_random_seed(worker_seed)


def wandb_logger(loss, metrics: Dict[str, float], epoch, mode: str) -> None:
    log_data = {"epoch": epoch}
    if loss:
        log_data[f"{mode}/loss/"] = loss

    for metric_name, metric_value in metrics.items():
        log_data[f"{mode}/metrics/{metric_name}"] = metric_value

    wandb.log(log_data)


def log_metrics_locally(
    all_metrics: Dict[str, Dict[str, float]],
    path_to_save: Path,
    epoch: int,
) -> None:
    """Summary table to the log and to metrics.csv.

    Per-class numbers are not written here: `Validator.save_extended_metrics` owns them,
    together with the threshold sweep, in extended_metrics.csv. Two files with the same
    long-format per-class rows only invites them to disagree.
    """
    metrics_df = pd.DataFrame.from_dict(all_metrics, orient="index")
    metrics_df = metrics_df.round(4)
    metrics_df = metrics_df[["accuracy", "f1", "precision", "recall"]]

    tabulated_data = tabulate(metrics_df, headers="keys", tablefmt="pretty", showindex=True)
    if epoch:
        logger.info(f"Metrics on epoch {epoch}:\n{tabulated_data}\n")
    else:
        logger.info(f"Best epoch metrics:\n{tabulated_data}\n")

    if path_to_save:
        metrics_df.to_csv(path_to_save / "metrics.csv")


def save_metrics(train_metrics, metrics, loss, epoch, path_to_save, use_wandb) -> None:
    log_metrics_locally(
        all_metrics={"train": train_metrics, "val": metrics},
        path_to_save=path_to_save,
        epoch=epoch,
    )
    if use_wandb:
        wandb_logger(loss, train_metrics, epoch, mode="train")
        wandb_logger(None, metrics, epoch, mode="val")


def calculate_remaining_time(
    one_epoch_time, epoch_start_time, epoch, epochs, cur_iter, all_iters
) -> str:
    if one_epoch_time is None:
        average_iter_time = (time.time() - epoch_start_time) / cur_iter
        remaining_iters = epochs * all_iters - cur_iter

        hours, remainder = divmod(average_iter_time * remaining_iters, 3600)
        minutes, _ = divmod(remainder, 60)
        return f"{int(hours):02}:{int(minutes):02}"

    time_for_remaining_epochs = one_epoch_time * (epochs + 1 - epoch)
    current_epoch_progress = time.time() - epoch_start_time
    hours, remainder = divmod(time_for_remaining_epochs - current_epoch_progress, 3600)
    minutes, _ = divmod(remainder, 60)
    return f"{int(hours):02}:{int(minutes):02}"


def get_vram_usage():
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,nounits,noheader",
            ],
            encoding="utf-8",
        )

        # Split lines to handle multiple GPUs correctly
        lines = output.strip().split("\n")
        total_usage = []

        for line in lines:
            try:
                used, total = map(float, line.split(", "))
                total_usage.append((used / total) * 100)
            except ValueError:
                print(f"Skipping malformed line: {line}")

        # If there are multiple GPUs, return the max usage percentage
        return round(max(total_usage)) if total_usage else 0

    except Exception as e:
        print(f"Error running nvidia-smi: {e}")
        return 0


def vis_one_image(image: np.ndarray, label: int, mode, label_to_name, score=None) -> None:
    if mode == "gt":
        prefix = "GT: "
        color = (46, 153, 60)
        postfix = ""
        position = (10, 30)
    elif mode == "pred":
        prefix = ""
        color = (148, 70, 44)
        postfix = f" {score:.2f}"
        position = (10, 50)

    cv2.putText(
        image,
        f"{prefix}{label_to_name.get(int(label), str(label))}{postfix}",
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2,
        cv2.LINE_AA,
    )


def visualize(img_paths, batch_gt, batch_probs, dataset_path, path_to_save, label_to_name):
    """
    Saves images with class names.
      - Green text for GT
      - Blue text for preds
    """
    path_to_save.mkdir(parents=True, exist_ok=True)

    for gt, prob, img_path in zip(batch_gt, batch_probs, img_paths):
        img = cv2.imread(str(dataset_path / img_path))

        pred = torch.argmax(prob).item()
        label = gt.item()
        score = prob.max().item()

        vis_one_image(img, label, mode="gt", label_to_name=label_to_name)
        vis_one_image(img, pred, mode="pred", label_to_name=label_to_name, score=score)

        # Construct a filename and save
        outpath = path_to_save / Path(img_path).name
        cv2.imwrite(str(outpath), np.ascontiguousarray(img))


class FocalLoss(nn.Module):
    """
    Focal Loss with optional label smoothing.

    Args:
        gamma (float): Focusing parameter gamma > 0 (default: 2.0).
        alpha (float or list or None): Class weighting factor. Can be a scalar
            (applied uniformly) or a list with length equal to the number of classes.
            If None, no class weighting is used.
        label_smoothing (float): Smoothing factor for label smoothing. A value in [0, 1)
            where 0 means no smoothing (default: 0.0).
        reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'
            (default: 'mean').
    """

    def __init__(self, gamma=2.0, alpha=None, label_smoothing=0.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

        # Process alpha: if provided as a scalar, convert to a tensor; if list, convert to tensor
        if alpha is not None:
            if isinstance(alpha, (float, int)):
                self.alpha = torch.tensor([alpha])
            elif isinstance(alpha, list):
                self.alpha = torch.tensor(alpha)
            else:
                raise TypeError("Alpha must be a float, int, or list.")
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        """
        Forward pass.

        Args:
            inputs (Tensor): Raw logits with shape (batch_size, num_classes).
            targets (Tensor): Ground truth class indices with shape (batch_size).

        Returns:
            Tensor: Loss value.
        """
        num_classes = inputs.size(1)
        # Compute log probabilities and probabilities
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)

        # Create one-hot encoding of targets
        with torch.no_grad():
            target_one_hot = torch.zeros_like(inputs).scatter(1, targets.unsqueeze(1), 1)
            if self.label_smoothing > 0:
                # Apply label smoothing:
                # For the true class: 1 - label_smoothing
                # For other classes: label_smoothing divided by (num_classes - 1)
                smooth_value = self.label_smoothing / (num_classes - 1)
                target_one_hot = target_one_hot * (1 - self.label_smoothing) + smooth_value

        # Compute the focal weight: (1 - p)^gamma.
        focal_weight = (1 - probs) ** self.gamma

        # If alpha is provided and is a tensor of length equal to number of classes,
        # then apply per-class weighting.
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            # If alpha has one element, treat it as a scalar factor.
            if self.alpha.numel() == 1:
                alpha_weight = self.alpha
            elif self.alpha.numel() == num_classes:
                # Reshape so that it broadcasts with the loss tensor.
                alpha_weight = self.alpha.view(1, -1)
            else:
                raise ValueError("Alpha length must be 1 or equal to number of classes.")
            focal_weight = alpha_weight * focal_weight

        # Compute the per-sample loss:
        loss = -target_one_hot * focal_weight * log_probs
        loss = loss.sum(dim=1)

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def get_latest_experiment_name(exp: str, output_dir: str):
    output_dir = Path(output_dir)
    if output_dir.exists():
        return exp

    target_exp_name = Path(exp).name.rsplit("_", 1)[0]
    latest_exp = None

    for exp_path in output_dir.parent.iterdir():
        exp_name, exp_date = exp_path.name.rsplit("_", 1)
        if target_exp_name == exp_name:
            exp_date = datetime.strptime(exp_date, "%Y-%m-%d")
            if not latest_exp or exp_date > latest_exp:
                latest_exp = exp_date

            print(target_exp_name, exp_date, latest_exp)

    final_exp_name = f"{target_exp_name}_{latest_exp.strftime('%Y-%m-%d')}"
    logger.info(f"Latest experiment: {final_exp_name}")
    return final_exp_name


def resolve_formats(requested, known, label: str) -> list:
    """null / missing -> every key in `known`; a list restricts it and rejects typos."""
    if not requested:
        return list(known)
    requested = [str(f).lower() for f in requested]
    unknown = set(requested) - set(known)
    if unknown:
        raise ValueError(f"unknown {label} {sorted(unknown)}; pick from {list(known)}")
    return requested


def auto_batch_size(
    cfg: DictConfig, num_labels: int, device: str, target_fraction: float = 0.7, default=32
) -> int:
    """Largest batch whose training step fits within `target_fraction` of total VRAM.

    Measures rather than estimates: a throwaway copy of the configured model runs a real
    forward + backward + AdamW step at each candidate size, under the run's AMP dtype and
    `layers_to_train` freezing, and the caching allocator's peak *reserved* bytes is what
    gets compared with the budget. Reserved, not allocated, because that is what the GPU
    is actually out of when training OOMs. The step matters: AdamW's two moment buffers
    (8 bytes per trainable weight) only appear on the first `optimizer.step()`, so a
    forward+backward-only probe would never see them. The EMA copy is held for the same
    reason when `use_ema` is on.

    Escalates through powers of two (1..1024), then binary-searches between the last size
    that fit and the first that did not. Input is synthetic - a classifier's footprint
    depends on shape alone - at the largest shape the train collate can emit.
    """
    dev = torch.device(device)
    if dev.type != "cuda":
        logger.warning(
            f"Auto batch size probes VRAM, so it only works on CUDA (got {device!r}); "
            f"using batch_size={default}"
        )
        return default

    logger.info("Searching for the optimal batch size...")
    total_mem = torch.cuda.get_device_properties(dev).total_memory
    target_mem = int(total_mem * target_fraction)
    dev_index = dev.index if dev.index is not None else torch.cuda.current_device()

    # fork_rng: timm's random init and the synthetic batches consume the torch RNG, and
    # the real model built afterwards must init exactly as it would with an explicit
    # batch_size, or "same seed" stops meaning "same run".
    with torch.random.fork_rng(devices=[dev_index]):
        best = _probe_batch_sizes(cfg, num_labels, device, target_mem)
    torch.cuda.empty_cache()  # the probe's frame is gone; hand its blocks back to CUDA

    total_gb = total_mem / 1024**3
    if best == 0:
        logger.warning(
            f"Even batch_size=1 exceeds {target_fraction:.0%} of {total_gb:.1f} GB VRAM; "
            "training with 1 and hoping the headroom covers it"
        )
        return 1
    logger.info(
        f"Optimal batch size: {best} (target {target_fraction:.0%} of {total_gb:.1f} GB VRAM)"
    )
    return best


def _probe_batch_sizes(cfg: DictConfig, num_labels: int, device: str, target_mem: int) -> int:
    """The search itself; 0 when not even batch 1 fits.

    Everything it allocates - model, EMA copy, optimizer state, the last probe's graph -
    is a local of this frame, so it is released as a unit on return.
    """
    dev = torch.device(device)
    h, w = (int(v) for v in cfg.train.img_size)
    if float(cfg.train.augs.multiscale_prob) > 0:
        h, w = h + 32, w + 32  # train_collate_fn's upscale is the largest batch it emits
    amp_enabled = bool(cfg.train.amp_enabled)
    amp_dtype = torch.float16 if cfg.train.get("amp_dtype") == "float16" else torch.bfloat16

    # Off: `build_model` would announce the frozen groups a second time for the copy.
    logger.disable("img_clf")
    try:
        model = build_model(
            model_name=cfg.model_name,
            num_labels=num_labels,
            pretrained=False,  # weights do not change the footprint, so skip the load
            device=device,
            layers_to_train=cfg.train.layers_to_train,
        )
    finally:
        logger.enable("img_clf")
    model.train()
    _ema = deepcopy(model) if cfg.train.use_ema else None  # held only for its VRAM
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
    loss_fn = nn.CrossEntropyLoss()

    def fits(bs: int) -> bool:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)
        try:
            inputs = torch.randn(bs, 3, h, w, device=dev)
            labels = torch.randint(0, num_labels, (bs,), device=dev)
            if amp_enabled:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    output = model(inputs)
            else:
                output = model(inputs)
            loss_fn(output.float(), labels).backward()
            # No GradScaler on purpose: a skipped fp16 step would leave the moment buffers
            # unallocated, and NaN weights cost the same bytes as finite ones.
            optimizer.step()
            return torch.cuda.max_memory_reserved(dev) <= target_mem
        except RuntimeError as e:
            # cuDNN / cuBLAS raise a plain RuntimeError for a failed workspace alloc.
            if isinstance(e, torch.OutOfMemoryError) or "out of memory" in str(e).lower():
                return False
            raise
        finally:
            optimizer.zero_grad(set_to_none=True)

    best, fail = 0, None
    for bs in (2**i for i in range(0, 11)):  # 1, 2, 4, ... 1024
        if fits(bs):
            best = bs
        else:
            fail = bs
            break

    if fail is not None and fail - best > 1:
        lo, hi = best + 1, fail - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if fits(mid):
                best, lo = mid, mid + 1
            else:
                hi = mid - 1
    return best
