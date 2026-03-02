from __future__ import annotations

import argparse
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from tqdm import tqdm

from dataset import build_dataloaders
from model import build_model


# ── Reproducibility ───────────────────────────────────────────────────────────

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── LR schedule: linear warmup → cosine annealing with warm restarts ─────────

class WarmupCosineAnnealingWarmRestarts:
    """
    Linear warmup for `warmup_epochs`, then CosineAnnealingWarmRestarts (SGDR).

    Cycle lengths:  T_0,  T_0 * T_mult,  T_0 * T_mult^2, ...
    At each restart the LR jumps back to `base_lr` and decays to `eta_min`.

    Args:
        optimizer      : wrapped optimiser
        warmup_epochs  : linear ramp-up duration (LR goes 0 → base_lr)
        T_0            : length of the first cosine cycle (epochs)
        T_mult         : factor applied to each successive cycle length (≥ 1)
        eta_min        : minimum LR at the trough of every cycle
        base_lr        : peak LR (= optimizer initial lr)
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        T_0: int,
        T_mult: int,
        eta_min: float,
        base_lr: float,
    ) -> None:
        self.optimizer     = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_lr       = base_lr
        self.current_epoch = 0

        self.cosine_scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=T_0,
            T_mult=T_mult,
            eta_min=eta_min,
        )

        # Start at zero so the first step() ramps up from ~0
        self._set_lr(0.0)

    def _set_lr(self, lr: float) -> None:
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def step(self) -> None:
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup: epoch 1 → base_lr/W, …, epoch W → base_lr
            warmup_lr = self.base_lr * self.current_epoch / self.warmup_epochs
            self._set_lr(warmup_lr)
        else:
            # Delegate to SGDR; shift epoch index to start at 0 after warmup
            cosine_epoch = self.current_epoch - self.warmup_epochs
            self.cosine_scheduler.step(cosine_epoch)

    def get_last_lr(self) -> list[float]:
        return [pg["lr"] for pg in self.optimizer.param_groups]

    def state_dict(self) -> dict:
        return {
            "current_epoch":    self.current_epoch,
            "cosine_scheduler": self.cosine_scheduler.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.current_epoch = state["current_epoch"]
        self.cosine_scheduler.load_state_dict(state["cosine_scheduler"])


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: DictConfig,
) -> WarmupCosineAnnealingWarmRestarts:
    sch = cfg.training.scheduler
    return WarmupCosineAnnealingWarmRestarts(
        optimizer=optimizer,
        warmup_epochs=sch.warmup_epochs,
        T_0=sch.T_0,
        T_mult=sch.T_mult,
        eta_min=sch.eta_min,
        base_lr=cfg.training.lr,
    )


# ── Cycle position helper ─────────────────────────────────────────────────────

def current_cycle_info(
    epoch: int, warmup: int, T_0: int, T_mult: int
) -> tuple[int, int, int]:
    """
    Returns (cycle_index, epoch_within_cycle, cycle_length) for a given
    absolute epoch (1-based), accounting for the warmup offset.
    """
    e = max(0, epoch - warmup)
    if e == 0:
        return 0, 0, T_0

    cycle  = 0
    length = T_0
    elapsed = 0
    while elapsed + length <= e:
        elapsed += length
        cycle   += 1
        length   = length * T_mult if T_mult > 1 else T_0

    return cycle, e - elapsed, length


# ── Metrics helper ────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_metrics(
    all_labels: list, all_preds: list, all_probs: list
) -> dict:
    labels = np.array(all_labels)
    preds  = np.array(all_preds)
    probs  = np.array(all_probs)

    metrics = {
        "accuracy":  accuracy_score(labels, preds),
        "f1":        f1_score(labels, preds, zero_division=0),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall":    recall_score(labels, preds, zero_division=0),
        "mcc":       matthews_corrcoef(labels, preds),
    }

    if len(set(labels)) == 2:
        metrics["auc_roc"] = roc_auc_score(labels, probs)
    else:
        metrics["auc_roc"] = float("nan")

    return metrics


# ── Gradient diagnostics ──────────────────────────────────────────────────────

def check_gradients(model: nn.Module, cfg: DictConfig) -> dict:
    grad_cfg    = cfg.diagnostics.gradient_check
    layer_norms: dict[str, float] = {}
    dead_layers: list[str]        = []
    total_sq    = 0.0

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            dead_layers.append(f"{name} (None)")
            continue
        norm = param.grad.data.norm(2).item()
        if norm == 0.0:
            dead_layers.append(f"{name} (zero)")
        layer_norms[name] = norm
        total_sq += norm ** 2

    global_norm = total_sq ** 0.5
    return {
        "global_norm": global_norm,
        "layer_norms": layer_norms,
        "dead_layers": dead_layers,
        "vanishing":   global_norm < grad_cfg.vanishing_threshold,
        "exploding":   global_norm > grad_cfg.exploding_threshold,
    }


def print_gradient_report(report: dict, verbose: bool = False) -> None:
    parts = [f"Global grad norm: {report['global_norm']:.4e}"]
    if report["vanishing"]:
        parts.append("⚠️  VANISHING GRADIENTS")
    if report["exploding"]:
        parts.append("⚠️  EXPLODING GRADIENTS")
    if report["dead_layers"]:
        parts.append(
            f"⚠️  Dead layers ({len(report['dead_layers'])}): "
            + ", ".join(report["dead_layers"][:5])
        )
    tqdm.write("  [GradCheck] " + " | ".join(parts))

    if verbose and report["layer_norms"]:
        tqdm.write("  [GradCheck] Per-layer norms:")
        for name, norm in report["layer_norms"].items():
            tqdm.write(f"             {name:<50s}  {norm:.4e}")


# ── Overfit diagnostics ───────────────────────────────────────────────────────

def check_overfitting(history: list[dict], cfg: DictConfig) -> dict | None:
    ov_cfg = cfg.diagnostics.overfit_check
    window = ov_cfg.window

    if len(history) < window:
        return None

    recent         = history[-window:]
    avg_train_loss = np.mean([h["train_loss"] for h in recent])
    avg_val_loss   = np.mean([h["val_loss"]   for h in recent])
    avg_train_acc  = np.mean([h["train_acc"]  for h in recent])
    avg_val_acc    = np.mean([h["val_acc"]    for h in recent])

    loss_gap = avg_val_loss  - avg_train_loss
    acc_gap  = avg_train_acc - avg_val_acc

    val_losses = [h["val_loss"] for h in recent]
    return {
        "avg_train_loss": avg_train_loss,
        "avg_val_loss":   avg_val_loss,
        "avg_train_acc":  avg_train_acc,
        "avg_val_acc":    avg_val_acc,
        "loss_gap":       loss_gap,
        "acc_gap":        acc_gap,
        "overfitting":    (
            loss_gap > ov_cfg.loss_gap_threshold or
            acc_gap  > ov_cfg.acc_gap_threshold
        ),
        "underfitting": avg_train_acc < ov_cfg.underfitting_acc_threshold,
        "rising_val":   val_losses[-1] > val_losses[0],
    }


def print_overfit_report(report: dict) -> None:
    tqdm.write(
        f"  [OverfitCheck] "
        f"train_loss={report['avg_train_loss']:.4f}  "
        f"val_loss={report['avg_val_loss']:.4f}  "
        f"Δloss={report['loss_gap']:+.4f}  |  "
        f"train_acc={report['avg_train_acc']:.3f}  "
        f"val_acc={report['avg_val_acc']:.3f}  "
        f"Δacc={report['acc_gap']:+.3f}"
    )
    flags = []
    if report["overfitting"]:
        flags.append("⚠️  OVERFITTING DETECTED")
    if report["underfitting"]:
        flags.append("⚠️  UNDERFITTING DETECTED")
    if report["rising_val"]:
        flags.append("📈 val_loss trending upward")
    if flags:
        tqdm.write("  [OverfitCheck] " + " | ".join(flags))


# ── Train / evaluate one epoch ────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> float:

    model.train()
    total_loss = 0.0
    n_batches  = 0
    
    pbar = tqdm(
        loader,
        desc=f"Epoch {epoch:>4}/{total_epochs} [train]",
        leave=False,
        dynamic_ncols=True,
        unit="batch",
    )

    for embeddings, labels, _pep_ids in pbar:
        embeddings = embeddings.to(device)
        labels     = labels.to(device)

        optimizer.zero_grad()
        logits = model(embeddings)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

        pbar.set_postfix(loss=f"{loss.item():.5f}", avg=f"{total_loss / n_batches:.5f}")

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    split_name: str = "val",
    epoch: int = 0,
    total_epochs: int = 0,
) -> tuple[float, dict]:
    model.eval()
    total_loss = 0.0
    n_batches  = 0
    all_labels = []
    all_preds  = []
    all_probs  = []

    pbar = tqdm(
        loader,
        desc=f"Epoch {epoch:>4}/{total_epochs} [{split_name:>5}]",
        leave=False,
        dynamic_ncols=True,
        unit="batch",
    )

    for embeddings, labels, _pep_ids in pbar:
        embeddings = embeddings.to(device)
        labels     = labels.to(device)

        logits = model(embeddings)
        loss   = criterion(logits, labels)

        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)

        total_loss += loss.item()
        n_batches  += 1
        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())

        pbar.set_postfix(loss=f"{loss.item():.5f}")

    avg_loss = total_loss / max(n_batches, 1)
    metrics  = compute_metrics(all_labels, all_preds, all_probs)
    return avg_loss, metrics


# ── Early stopping tracker ────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best_loss = float("inf")

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineAnnealingWarmRestarts,
    epoch: int,
    val_loss: float,
    cfg: DictConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch":       epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "sched_state": scheduler.state_dict(),
            "val_loss":    val_loss,
            "config":      OmegaConf.to_container(cfg, resolve=True),
        },
        path,
    )


# ── Main training loop ───────────────────────────────────────────────────────

def train(cfg: DictConfig) -> None:
    seed_everything(cfg.training.seed)

    device = torch.device(
        cfg.training.device if torch.cuda.is_available() else "cpu"
    )
    print(f"[train] Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = build_dataloaders(cfg)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(cfg).to(device)

    # ── Loss ──────────────────────────────────────────────────────────────────
    weight = None
    if cfg.training.use_class_weights:
        weight = train_loader.dataset.class_weights.to(device)
        print(f"[train] Class weights: {weight.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=weight)

    # ── Optimiser + scheduler ─────────────────────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = build_scheduler(optimizer, cfg)
    sch_cfg   = cfg.training.scheduler

    print(
        f"[train] Scheduler: {sch_cfg.warmup_epochs}-epoch linear warmup → "
        f"CosineAnnealingWarmRestarts "
        f"(T_0={sch_cfg.T_0}, T_mult={sch_cfg.T_mult}, eta_min={sch_cfg.eta_min:.1e})"
    )

    # ── Early stopping ────────────────────────────────────────────────────────
    es_cfg = cfg.training.early_stopping
    early_stopper = (
        EarlyStopping(patience=es_cfg.patience, min_delta=es_cfg.min_delta)
        if es_cfg.enabled
        else None
    )

    # ── Diagnostics ───────────────────────────────────────────────────────────
    diag_cfg      = cfg.diagnostics
    grad_every    = diag_cfg.gradient_check.every_n_epochs
    overfit_every = diag_cfg.overfit_check.every_n_epochs

    # ── Checkpointing ────────────────────────────────────────────────────────
    ckpt_dir      = Path(cfg.training.checkpoint.dir)
    best_path     = ckpt_dir / "best.pt"
    last_path     = ckpt_dir / "last.pt"
    best_val_loss = float("inf")

    history: list[dict] = []

    # ── Epoch progress bar ────────────────────────────────────────────────────
    epoch_pbar = tqdm(
        range(1, cfg.training.epochs + 1),
        desc="Training",
        unit="epoch",
        dynamic_ncols=True,
    )

    print("\n══════════════════════════════════════════════════════════════")
    print("  Training starts")
    print("══════════════════════════════════════════════════════════════\n")

    header = (
        f"{'Ep':>4}  │ {'Train Loss':>10}  │ {'Val Loss':>10}  │ "
        f"{'Acc':>6}  {'F1':>6}  {'MCC':>6}  {'AUC':>6}  │ "
        f"{'LR':>10}  {'Cycle':>14}  │ Note"
    )
    print(header)
    print("─" * len(header))

    for epoch in epoch_pbar:

        # ── Train ─────────────────────────────────────────────────────────────
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch, cfg.training.epochs,
        )

        # ── Train-set eval (for overfit tracking) ─────────────────────────────
        train_loss_eval, train_metrics_eval = evaluate(
            model, train_loader, criterion, device,
            split_name="train", epoch=epoch, total_epochs=cfg.training.epochs,
        )

        # ── Validate ──────────────────────────────────────────────────────────
        val_loss, val_metrics = evaluate(
            model, val_loader, criterion, device,
            split_name="val", epoch=epoch, total_epochs=cfg.training.epochs,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # ── Cycle position ────────────────────────────────────────────────────
        cycle_idx, epoch_in_cycle, cycle_len = current_cycle_info(
            epoch,
            warmup=sch_cfg.warmup_epochs,
            T_0=sch_cfg.T_0,
            T_mult=sch_cfg.T_mult,
        )
        # Detect restart: first epoch of a new cycle (but not the very first cycle)
        is_restart = (
            epoch > sch_cfg.warmup_epochs
            and epoch_in_cycle == 1
            and cycle_idx > 0
        )
        cycle_label = f"C{cycle_idx} {epoch_in_cycle:>3}/{cycle_len}"

        # ── History ───────────────────────────────────────────────────────────
        history.append({
            "train_loss": train_loss_eval,
            "val_loss":   val_loss,
            "train_acc":  train_metrics_eval["accuracy"],
            "val_acc":    val_metrics["accuracy"],
        })

        # ── Checkpointing ─────────────────────────────────────────────────────
        note = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if cfg.training.checkpoint.save_best:
                save_checkpoint(best_path, model, optimizer, scheduler, epoch, val_loss, cfg)
                note = "★ best"

        if cfg.training.checkpoint.save_last:
            save_checkpoint(last_path, model, optimizer, scheduler, epoch, val_loss, cfg)

        if is_restart:
            note = ("↺ restart  " + note).strip()

        # ── Log ───────────────────────────────────────────────────────────────
        tqdm.write(
            f"{epoch:4d}  │ {train_loss:10.5f}  │ {val_loss:10.5f}  │ "
            f"{val_metrics['accuracy']:6.3f}  "
            f"{val_metrics['f1']:6.3f}  "
            f"{val_metrics['mcc']:6.3f}  "
            f"{val_metrics['auc_roc']:6.3f}  │ "
            f"{current_lr:10.2e}  {cycle_label:>14}  │ {note}"
        )

        epoch_pbar.set_postfix(
            train=f"{train_loss:.4f}",
            val=f"{val_loss:.4f}",
            acc=f"{val_metrics['accuracy']:.3f}",
            f1=f"{val_metrics['f1']:.3f}",
            cycle=f"C{cycle_idx}",
        )

        # ── Gradient check ────────────────────────────────────────────────────
        if grad_every > 0 and epoch % grad_every == 0:
            model.train()
            optimizer.zero_grad()
            embeddings, labels, _ = next(iter(train_loader))
            logits = model(embeddings.to(device))
            criterion(logits, labels.to(device)).backward()

            grad_report = check_gradients(model, cfg)
            optimizer.zero_grad()

            tqdm.write(f"\n  ── Gradient check @ epoch {epoch} ──")
            print_gradient_report(grad_report, verbose=diag_cfg.gradient_check.verbose)
            tqdm.write("")

        # ── Overfit check ─────────────────────────────────────────────────────
        if overfit_every > 0 and epoch % overfit_every == 0:
            ov_report = check_overfitting(history, cfg)
            if ov_report is not None:
                tqdm.write(f"\n  ── Overfit check @ epoch {epoch} ──")
                print_overfit_report(ov_report)
                tqdm.write("")

        # ── Early stopping ────────────────────────────────────────────────────
        if early_stopper is not None and early_stopper.step(val_loss):
            tqdm.write(
                f"\n[early stopping] No improvement for {es_cfg.patience} epochs. "
                f"Stopping at epoch {epoch}."
            )
            break

    epoch_pbar.close()

    # ── Final test evaluation ─────────────────────────────────────────────────
    print("\n══════════════════════════════════════════════════════════════")
    print("  Test-set evaluation  (loading best checkpoint)")
    print("══════════════════════════════════════════════════════════════\n")

    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        print(f"  Loaded best checkpoint from epoch {ckpt['epoch']} "
              f"(val_loss={ckpt['val_loss']:.5f})\n")
    else:
        print("  (No best checkpoint found — evaluating last model state)\n")

    test_loss, test_metrics = evaluate(
        model, test_loader, criterion, device,
        split_name="test", epoch=0, total_epochs=0,
    )

    print(f"  Test loss : {test_loss:.5f}")
    for k, v in test_metrics.items():
        print(f"  {k:<10}: {v:.4f}")

    # ── Classification report ─────────────────────────────────────────────────
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for embeddings, labels, _ in tqdm(test_loader, desc="Final inference", leave=False):
            preds = model(embeddings.to(device)).argmax(dim=1)
            all_labels.extend(labels.tolist())
            all_preds.extend(preds.cpu().tolist())

    print("\n  Classification report")
    print("  " + "─" * 52)
    report = classification_report(
        all_labels, all_preds, target_names=["Inactive", "Active"], digits=4
    )
    for line in report.split("\n"):
        print(f"  {line}")

    cm = confusion_matrix(all_labels, all_preds)
    print(f"\n  Confusion matrix:\n{cm}")
    print("\n  Done ✓")


# ── CLI entry-point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PeptideTransformer")
    parser.add_argument(
        "--config",
        default="configs/train_config.yaml",
        help="Path to config YAML",
    )
    args = parser.parse_args()
    cfg  = OmegaConf.load(args.config)
    train(cfg)