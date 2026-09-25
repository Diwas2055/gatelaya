#!/usr/bin/env python3
"""Fine-tune Laya on the GateLaya held-out splits (official algorithm, single device).

Implements the reference notebook's training loop (evals/reference/
laya_finetune_official.ipynb cells 6+8) without DDP: proper_reward RLCD with
eps-noise GROUP_SIZE, soft cross-entropy guidance, AdamW + cosine, gradient
checkpointing, rolling best-val-accuracy checkpoint. Trains only the production noul
question per row with target [1-label, label].

Device: MPS fp32 when available (CPU fallback); on an MPS kernel failure the
whole run is retried once on CPU. The saved checkpoint ships a NEUTRAL
temperature ([1.0, 1.0, 1.0]) — calibrate on val afterwards.

Smoke:  uv run scripts/finetune.py --epochs 1 --limit 48 --out /tmp/ft-smoke
Full:   uv run scripts/finetune.py --epochs 4 --out evals/models/gatelaya-ft
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any

try:
    from gatelaya.questions import (
        INJECTION_INSTRUCTIONS,
        PII_INSTRUCTIONS,
        SECRET_LEAK_INSTRUCTIONS,
        TOXICITY_INSTRUCTIONS,
    )
except ModuleNotFoundError:  # running from a source checkout without install
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from gatelaya.questions import (
        INJECTION_INSTRUCTIONS,
        PII_INSTRUCTIONS,
        SECRET_LEAK_INSTRUCTIONS,
        TOXICITY_INSTRUCTIONS,
    )

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLITS = ROOT / "evals" / "data" / "splits"
DEFAULT_LOG = ROOT / "evals" / "results" / "finetune-log.json"

CHECK_INSTRUCTIONS = {
    "pii": PII_INSTRUCTIONS,
    "injection": INJECTION_INSTRUCTIONS,
    "toxicity": TOXICITY_INSTRUCTIONS,
    "secret_leak": SECRET_LEAK_INSTRUCTIONS,
}

# Hyperparameters (official algorithm + project spec).
MICRO_BATCH = 8
GRAD_ACCUM = 4
GROUP_SIZE = 4
W_SPH = 0.75
LR_DEFAULT = 2e-5  # encoder; head gets 4x
LR_HEAD_MULT = 4.0
WEIGHT_DECAY = 0.01
SIGMA_START = 0.4
SIGMA_END = 0.1
CLIP_NORM = 1.0
PATIENCE_DEFAULT = 2
MAX_LEN = 1024
HEAD_MAX_LEN = 256


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="convaiinnovations/laya-multilingual")
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--out", type=Path, default=ROOT / "evals" / "models" / "gatelaya-ft")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="cap train rows (smoke runs)")
    parser.add_argument("--lr", type=float, default=LR_DEFAULT, help="encoder LR (head = 4x)")
    parser.add_argument("--patience", type=int, default=PATIENCE_DEFAULT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    return parser.parse_args(argv)


def resolve_device(choice: str) -> str:
    """Pick the torch device string (auto = mps if available, else cpu)."""
    import torch

    if choice != "auto":
        return choice
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_split(path: Path) -> list[dict[str, Any]]:
    """Read one JSONL split."""
    if not path.exists():
        raise SystemExit(f"missing split file: {path} (run scripts/prepare_data.py first)")
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build_items(
    rows: list[dict[str, Any]], tok: Any, cfg: dict[str, Any]
) -> tuple[list[dict[str, Any]], int]:
    """Build noul training items from dataset rows; returns (items, skipped)."""
    from laya.common import QTYPES, build_sequence, render_options

    items: list[dict[str, Any]] = []
    skipped = 0
    for row in rows:
        instruction = CHECK_INSTRUCTIONS[row["check"]]
        q = {"t": "noul", "ins": instruction, "crit": {}}
        k = len(render_options(q))
        seq, markers = build_sequence(
            tok, {"text": row["text"]}, q, cfg["max_len"], cfg["head_max_len"]
        )
        if len(markers) != k:
            skipped += 1
            continue
        label = int(row["label"])
        items.append(
            {
                "ids": seq,
                "markers": markers,
                "qtype": QTYPES["noul"],
                "target": [1.0 - float(label), float(label)],
                "label": label,
            }
        )
    return items, skipped


def collate_train_batch(items: list[dict[str, Any]], pad_id: int) -> dict[str, Any]:
    """Pad a micro-batch (verbatim from the official notebook)."""
    import torch

    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        target[i, : len(it["target"])] = torch.tensor(it["target"], dtype=torch.float32)
    return {
        "input_ids": ids,
        "attention_mask": att,
        "marker_pos": mpos,
        "marker_mask": mmask,
        "target": target,
        "qtype": torch.tensor([it["qtype"] for it in items]),
        "label": torch.tensor([it["label"] for it in items]),
    }


def evaluate(model: Any, items: list[dict[str, Any]], device: str, pad_id: int) -> dict[str, float]:
    """Val loss (soft CE) + hard accuracy, no grad, eval mode."""
    import torch

    if not items:
        return {"loss": float("nan"), "accuracy": float("nan")}
    was_training = model.training
    model.eval()
    total_loss, correct, seen = 0.0, 0, 0
    with torch.no_grad():
        for start in range(0, len(items), 16):
            chunk = items[start : start + 16]
            batch = collate_train_batch(chunk, pad_id)
            logits, _act = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["marker_pos"].to(device),
                batch["marker_mask"].to(device),
                batch["qtype"].to(device),
            )
            mask = batch["marker_mask"].to(device)
            target = batch["target"].to(device)
            loss = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1)
            total_loss += float(loss.mean()) * len(chunk)
            preds = logits.masked_fill(~mask, -1e4).argmax(-1)
            correct += int((preds == batch["label"].to(device)).sum())
            seen += len(chunk)
    if was_training:
        model.train()
    return {"loss": total_loss / max(1, seen), "accuracy": correct / max(1, seen)}


def save_checkpoint(model: Any, tok: Any, cfg: dict[str, Any], out_dir: Path) -> None:
    """Save laya-loadable checkpoint: fp16 weights + encoder + tokenizer + config (neutral temp)."""
    import json as _json

    from safetensors.torch import save_file

    out_dir.mkdir(parents=True, exist_ok=True)
    state = {k: v.half().contiguous().cpu() for k, v in model.state_dict().items()}
    save_file(state, str(out_dir / "model.safetensors"))
    model.encoder.config.save_pretrained(str(out_dir / "encoder"))
    tok.save_pretrained(str(out_dir / "tokenizer"))
    out_cfg = dict(cfg)
    out_cfg["fine_tuned"] = True
    out_cfg["model_name"] = "laya-gatelaya-ft"
    out_cfg["temperature"] = [1.0, 1.0, 1.0]  # neutral; fit on val afterwards
    out_cfg.pop("temperature_by_options", None)
    with (out_dir / "rl_agent_config.json").open("w", encoding="utf-8") as fh:
        _json.dump(out_cfg, fh, indent=2)


def train_run(
    base: str,
    device: str,
    train_items: list[dict[str, Any]],
    val_items: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Load base checkpoint, train with the official algorithm, save best. Returns log dict."""
    import laya
    import torch
    from laya.common import proper_reward

    agent = laya.load(base, device=device)
    tok = agent.tok
    model = agent.model
    cfg = dict(agent.cfg)
    cfg["gradient_checkpointing"] = True
    cfg["max_tokens_per_batch"] = 4096
    cfg["max_len"] = MAX_LEN
    cfg["head_max_len"] = HEAD_MAX_LEN

    try:
        model.encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    except Exception as exc:  # pragma: no cover - encoder-dependent
        print(f"warning: gradient checkpointing unavailable ({exc}); continuing without")
    model.head_checkpointing = True
    model.to(device)
    model.train()

    enc_params = [p for n, p in model.named_parameters() if "encoder." in n]
    head_params = [p for n, p in model.named_parameters() if "encoder." not in n]
    optimizer = torch.optim.AdamW(
        [
            {"params": enc_params, "lr": args.lr},
            {"params": head_params, "lr": args.lr * LR_HEAD_MULT},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    total_updates = (len(train_items) // (MICRO_BATCH * GRAD_ACCUM)) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_updates), eta_min=1e-6
    )

    epoch_logs: list[dict[str, Any]] = []
    best_key = (-1.0, float("-inf"))  # (val_accuracy, -val_loss) rank for best-epoch pick
    best_val = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    bad_epochs = 0
    stopped_early = False
    t0 = time.time()
    pad_id = tok.pad_token_id

    for epoch in range(args.epochs):
        random.seed(args.seed + epoch)
        random.shuffle(train_items)
        epoch_loss, n_batches = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        accum = 0
        progress = epoch / max(1, args.epochs - 1)
        sigma = SIGMA_START + (SIGMA_END - SIGMA_START) * progress

        for b_idx in range(0, len(train_items), MICRO_BATCH):
            chunk = train_items[b_idx : b_idx + MICRO_BATCH]
            if not chunk:
                continue
            batch = collate_train_batch(chunk, pad_id)
            logits, act = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["marker_pos"].to(device),
                batch["marker_mask"].to(device),
                batch["qtype"].to(device),
            )
            logits = logits.float()
            mask = batch["marker_mask"].to(device)
            k = mask.sum(-1, keepdim=True).float()
            target = batch["target"].to(device)

            # 1. Sample GROUP_SIZE noisy logit distributions, zero-mean projected.
            eps = torch.randn((GROUP_SIZE,) + logits.shape, device=device) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask, -1e4), -1)

            # 2. Proper scoring reward + advantage normalisation.
            with torch.no_grad():
                r = proper_reward(
                    q, target.unsqueeze(0), batch["qtype"].to(device), mask, w_sph=W_SPH, w_rps=1.0
                )
                adv = r - r.mean(0, keepdim=True)
                adv = adv / (adv.std() + 1e-6)

            # 3. Policy gradient + full soft cross-entropy guidance.
            logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma**2)
            loss_rl = -(adv * logp).mean()
            loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
            loss = (loss_rl + 1.0 * loss_ce) / GRAD_ACCUM + 0.0 * act.sum()

            loss.backward()
            accum += 1
            if accum % GRAD_ACCUM == 0 or (b_idx + MICRO_BATCH) >= len(train_items):
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += float(loss.detach()) * GRAD_ACCUM
            n_batches += 1

        val = evaluate(model, val_items, device, pad_id)
        train_avg = epoch_loss / max(1, n_batches)
        entry = {
            "epoch": epoch + 1,
            "train_loss": round(train_avg, 4),
            "val_loss": round(val["loss"], 4),
            "val_accuracy": round(val["accuracy"], 4),
            "lr": scheduler.get_last_lr()[0],
            "sigma": round(sigma, 3),
            "seconds": round(time.time() - t0, 1),
        }
        epoch_logs.append(entry)
        print(
            f"epoch {epoch + 1}/{args.epochs} | train {train_avg:.4f} | "
            f"val loss {val['loss']:.4f} | val acc {val['accuracy']:.4f} | "
            f"lr {entry['lr']:.2e} | sigma {sigma:.2f} | {entry['seconds']}s"
        )

        # Gate is macro accuracy: select best epoch by val accuracy,
        # tie-break on val loss (calibration proxy).
        key = (val["accuracy"], -val["loss"])
        if key > best_key:
            best_key = key
            best_val = val["loss"]
            best_val_acc = val["accuracy"]
            best_epoch = epoch + 1
            bad_epochs = 0
            save_checkpoint(model, tok, cfg, args.out)
            print(f"  saved best checkpoint -> {args.out}")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                stopped_early = True
                print(f"  early stop (no val improvement for {args.patience} epochs)")
                break

    return {
        "base": base,
        "device": device,
        "hyperparameters": {
            "epochs": args.epochs,
            "lr_encoder": args.lr,
            "lr_head": args.lr * LR_HEAD_MULT,
            "micro_batch": MICRO_BATCH,
            "grad_accum": GRAD_ACCUM,
            "group_size": GROUP_SIZE,
            "w_sph": W_SPH,
            "weight_decay": WEIGHT_DECAY,
            "sigma": [SIGMA_START, SIGMA_END],
            "clip_norm": CLIP_NORM,
            "patience": args.patience,
            "max_len": MAX_LEN,
            "head_max_len": HEAD_MAX_LEN,
            "seed": args.seed,
            "temperature_saved": [1.0, 1.0, 1.0],
        },
        "train_items": len(train_items),
        "val_items": len(val_items),
        "epochs_run": len(epoch_logs),
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val, 4),
        "best_val_accuracy": round(best_val_acc, 4),
        "selection": "max val accuracy, tie-break min val loss",
        "stopped_early": stopped_early,
        "epoch_logs": epoch_logs,
        "total_seconds": round(time.time() - t0, 1),
        "output": str(args.out),
    }


def main(argv: list[str] | None = None) -> int:
    """Build items, train (MPS with one CPU retry), write the run log."""
    args = parse_args(argv)
    device = resolve_device(args.device)

    # Build items with the base tokenizer/config before touching the trainer.
    import laya

    print(f"loading base checkpoint {args.base!r} ...")
    probe = laya.load(args.base, device="cpu")
    cfg = dict(probe.cfg)
    cfg["max_len"] = MAX_LEN
    cfg["head_max_len"] = HEAD_MAX_LEN

    train_rows = load_split(args.splits / "train.jsonl")
    val_rows = load_split(args.splits / "val.jsonl")
    if args.limit > 0:
        train_rows = train_rows[: args.limit]
    train_items, train_skipped = build_items(train_rows, probe.tok, cfg)
    val_items, val_skipped = build_items(val_rows, probe.tok, cfg)
    print(
        f"train items {len(train_items)} (skipped {train_skipped}) | "
        f"val items {len(val_items)} (skipped {val_skipped})"
    )
    if not train_items or not val_items:
        raise SystemExit("no training/val items built — check splits and instructions")
    del probe

    def run_on(dev: str) -> dict[str, Any]:
        return train_run(args.base, dev, train_items, val_items, args)

    log: dict[str, Any]
    if device == "mps":
        try:
            log = run_on("mps")
        except Exception as exc:
            print(
                f"warning: MPS run failed ({type(exc).__name__}: {exc}); "
                "retrying the whole run on CPU",
                flush=True,
            )
            traceback.print_exc()
            log = run_on("cpu")
            log["mps_failure"] = f"{type(exc).__name__}: {exc}"
    else:
        log = run_on(device)

    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("w", encoding="utf-8") as fh:
        json.dump(log, fh, indent=2)
    print(f"run log -> {args.log}")
    print(
        f"best epoch {log['best_epoch']} | val acc {log['best_val_accuracy']} "
        f"| val loss {log['best_val_loss']} -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
