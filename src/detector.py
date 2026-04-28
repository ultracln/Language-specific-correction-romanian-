import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent))

from utils import (
    ERROR_TYPES,
    ID_TO_ERROR_TYPE,
    align_to_subwords,
    read_jsonl,
    set_seed,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data/prepared")
    p.add_argument("--out_dir", type=str, default="results/detector")
    p.add_argument("--model_name", type=str, default="readerbench/RoBERT-large")
    p.add_argument("--max_length", type=int, default=192)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.005)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--type_loss_weight", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--max_train_examples", type=int, default=-1)
    return p.parse_args()


class DetectorDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length, limit=-1):
        self.rows = list(read_jsonl(jsonl_path))
        if limit > 0:
            self.rows = self.rows[:limit]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        enc = align_to_subwords(row["tokens"], row["labels"], self.tokenizer, self.max_length)
        type_enc = align_to_subwords(row["tokens"], row["type_labels"], self.tokenizer, self.max_length)
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "det_labels": enc["labels"],
            "type_labels": type_enc["labels"],
        }


def collate(batch, pad_id):
    max_len = max(len(b["input_ids"]) for b in batch)
    out = {k: [] for k in ["input_ids", "attention_mask", "det_labels", "type_labels"]}
    for b in batch:
        pad = max_len - len(b["input_ids"])
        out["input_ids"].append(b["input_ids"] + [pad_id] * pad)
        out["attention_mask"].append(b["attention_mask"] + [0] * pad)
        out["det_labels"].append(b["det_labels"] + [-100] * pad)
        out["type_labels"].append(b["type_labels"] + [-100] * pad)
    return {k: torch.tensor(v) for k, v in out.items()}


class TwoHeadDetector(nn.Module):
    def __init__(self, model_name: str, num_types: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.det_head = nn.Linear(hidden, 2)
        self.type_head = nn.Linear(hidden, num_types)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = self.dropout(out.last_hidden_state)
        return self.det_head(h), self.type_head(h)


def evaluate(model, loader, device, type_weight):
    model.eval()
    det_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    type_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    total_loss = 0.0
    det_tp = det_fp = det_fn = det_tn = 0
    type_correct = type_total = 0

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            det_logits, type_logits = model(batch["input_ids"], batch["attention_mask"])
            det_loss = det_loss_fn(det_logits.view(-1, 2), batch["det_labels"].view(-1))
            type_loss = type_loss_fn(type_logits.view(-1, len(ERROR_TYPES)), batch["type_labels"].view(-1))
            total_loss += (det_loss + type_weight * type_loss).item()

            det_pred = det_logits.argmax(-1)
            det_mask = batch["det_labels"] != -100
            det_true = batch["det_labels"][det_mask]
            det_p = det_pred[det_mask]
            det_tp += ((det_p == 1) & (det_true == 1)).sum().item()
            det_fp += ((det_p == 1) & (det_true == 0)).sum().item()
            det_fn += ((det_p == 0) & (det_true == 1)).sum().item()
            det_tn += ((det_p == 0) & (det_true == 0)).sum().item()

            type_pred = type_logits.argmax(-1)
            type_mask = (batch["type_labels"] != -100) & (batch["type_labels"] != 0)
            if type_mask.any():
                type_correct += (type_pred[type_mask] == batch["type_labels"][type_mask]).sum().item()
                type_total += type_mask.sum().item()

    prec = det_tp / (det_tp + det_fp) if (det_tp + det_fp) else 0.0
    rec = det_tp / (det_tp + det_fn) if (det_tp + det_fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    f05 = (1.25 * prec * rec / (0.25 * prec + rec)) if (0.25 * prec + rec) else 0.0
    type_acc = type_correct / type_total if type_total else 0.0

    return {
        "loss": total_loss / max(len(loader), 1),
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "f05": f05,
        "type_acc": type_acc,
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    print(f"loading tokenizer + model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = TwoHeadDetector(args.model_name, num_types=len(ERROR_TYPES)).to(device)

    train_ds = DetectorDataset(Path(args.data_dir) / "detector_train.jsonl", tokenizer, args.max_length, args.max_train_examples)
    val_ds = DetectorDataset(Path(args.data_dir) / "detector_val.jsonl", tokenizer, args.max_length)
    print(f"train={len(train_ds)} val={len(val_ds)}")

    pad_id = tokenizer.pad_token_id
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=lambda b: collate(b, pad_id))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=lambda b: collate(b, pad_id))

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = (len(train_loader) // args.grad_accum) * args.epochs
    sched = get_linear_schedule_with_warmup(optim, int(args.warmup_ratio * total_steps), total_steps)

    det_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    type_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    best_f05 = -1.0
    history = []

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        optim.zero_grad()

        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}")
        for step, batch in enumerate(pbar):
            batch = {k: v.to(device) for k, v in batch.items()}
            det_logits, type_logits = model(batch["input_ids"], batch["attention_mask"])
            det_loss = det_loss_fn(det_logits.view(-1, 2), batch["det_labels"].view(-1))
            type_loss = type_loss_fn(type_logits.view(-1, len(ERROR_TYPES)), batch["type_labels"].view(-1))
            loss = (det_loss + args.type_loss_weight * type_loss) / args.grad_accum
            loss.backward()
            running += loss.item() * args.grad_accum

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                sched.step()
                optim.zero_grad()

            if step % 50 == 0:
                pbar.set_postfix({"loss": f"{running/(step+1):.4f}"})

        metrics = evaluate(model, val_loader, device, args.type_loss_weight)
        metrics["epoch"] = epoch + 1
        metrics["train_loss"] = running / len(train_loader)
        history.append(metrics)
        print(f"  val: {json.dumps(metrics, indent=2)}")

        if metrics["f05"] > best_f05:
            best_f05 = metrics["f05"]
            torch.save({
                "state_dict": model.state_dict(),
                "model_name": args.model_name,
                "args": vars(args),
            }, out_dir / "best.pt")
            print(f"  saved best (f0.5={best_f05:.4f})")

    with (out_dir / "history.json").open("w") as f:
        json.dump(history, f, indent=2)
    tokenizer.save_pretrained(out_dir / "tokenizer")
    print(f"done. best f0.5={best_f05:.4f}. artifacts in {out_dir}")


if __name__ == "__main__":
    main()
