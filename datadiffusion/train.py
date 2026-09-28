"""Training: epsilon-prediction MSE with uniform timesteps, AdamW + warmup/cosine, EMA weights.

v2 bugs this fixes: OneCycleLR was sized for 4x the real optimizer steps (with gradient
accumulation it never annealed), and timesteps were drawn ~1/(t+1) without reweighting, which
biased the objective toward t=0. Here every batch is one optimizer step and t ~ U{0..T-1}.
"""

from __future__ import annotations

import copy
import logging
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from datadiffusion.config import ExperimentConfig
from datadiffusion.data import Preprocessor, TabularData
from datadiffusion.model import TabularMLP
from datadiffusion.runs import MetricsWriter, save_checkpoint
from datadiffusion.schedule import NoiseSchedule

logger = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def warmup_cosine(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)


@torch.no_grad()
def _val_loss(model, schedule, x, t, noise) -> float:
    model.eval()
    loss = torch.mean((model(schedule.add_noise(x, noise, t), t) - noise) ** 2).item()
    model.train()
    return loss


def train(cfg: ExperimentConfig, data: TabularData, run: Path) -> dict:
    """Train on data.train, keep the best EMA weights by validation loss, save to run/model.pt."""
    tc = cfg.train
    seed_everything(tc.seed)
    X = torch.from_numpy(Preprocessor(cfg.data.seed).fit(data.train).transform(data.train))
    perm = torch.randperm(len(X), generator=torch.Generator().manual_seed(tc.seed))
    n_val = max(1, round(len(X) * tc.val_fraction))
    x_val, x_train = X[perm[:n_val]], X[perm[n_val:]]

    d = cfg.diffusion
    schedule = NoiseSchedule(d.num_timesteps, d.schedule, d.beta_start, d.beta_end)
    model = TabularMLP(X.shape[1], cfg.model)
    loader = DataLoader(
        TensorDataset(x_train),
        batch_size=tc.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(tc.seed),
    )
    total = tc.epochs * len(loader)
    opt = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: warmup_cosine(s, min(tc.warmup_steps, total), total)
    )
    ema = EMA(model, tc.ema_decay)

    # a fixed noise draw for validation makes epoch losses comparable
    g = torch.Generator().manual_seed(tc.seed + 1)
    val_t = torch.randint(0, d.num_timesteps, (len(x_val),), generator=g)
    val_noise = torch.randn(x_val.shape, generator=g)

    writer = MetricsWriter(run / "metrics.jsonl")
    best, best_epoch, best_state, stale = float("inf"), 0, None, 0
    eval_model = copy.deepcopy(model)
    logger.info(
        "training %d params on %d rows (%d val), %d steps",
        sum(p.numel() for p in model.parameters()),
        len(x_train),
        len(x_val),
        total,
    )
    for epoch in range(1, tc.epochs + 1):
        losses = []
        for (x,) in loader:
            t = torch.randint(0, d.num_timesteps, (len(x),))
            noise = torch.randn_like(x)
            loss = torch.mean((model(schedule.add_noise(x, noise, t), t) - noise) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
            opt.step()
            sched.step()
            ema.update(model)
            losses.append(loss.item())
        eval_model.load_state_dict(ema.shadow)
        val = _val_loss(eval_model, schedule, x_val, val_t, val_noise)
        writer.log(epoch=epoch, train_loss=float(np.mean(losses)), val_loss=val, lr=sched.get_last_lr()[0])
        if epoch == 1 or epoch % 10 == 0:
            logger.info(
                "epoch %d: train %.4f  val %.4f  lr %.2e", epoch, np.mean(losses), val, sched.get_last_lr()[0]
            )
        if val < best:
            best, best_epoch, stale = val, epoch, 0
            best_state = {k: v.clone() for k, v in ema.shadow.items()}
        else:
            stale += 1
            if stale >= tc.patience:
                logger.info("early stop at epoch %d (best %d)", epoch, best_epoch)
                break
    writer.close()
    model.load_state_dict(best_state)
    save_checkpoint(run, cfg, model, best_epoch=best_epoch, best_val_loss=best)
    return {"best_epoch": best_epoch, "best_val_loss": best, "epochs_run": epoch}
