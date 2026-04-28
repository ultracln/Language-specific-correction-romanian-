import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent))

from utils import read_jsonl, set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data/prepared")
    p.add_argument("--out_dir", type=str, default="results/seq2seq")
    p.add_argument("--model_name", type=str, default="google/mt5-small")
    p.add_argument("--max_source_length", type=int, default=192)
    p.add_argument("--max_target_length", type=int, default=192)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.005)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--exclude_error_types", nargs="+", default=[])
    p.add_argument("--max_train_examples", type=int, default=-1)
    p.add_argument("--beam_size", type=int, default=4)
    return p.parse_args()


class CorrectorDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, src_len, tgt_len, exclude=None, limit=-1):
        rows = list(read_jsonl(jsonl_path))
        if exclude:
            rows = [r for r in rows if r["error_type"] not in exclude]
        if limit > 0:
            rows = rows[:limit]
        self.rows = rows
        self.tok = tokenizer
        self.src_len = src_len
        self.tgt_len = tgt_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        src = self.tok(row["input"], max_length=self.src_len, truncation=True, padding=False)
        tgt = self.tok(row["target"], max_length=self.tgt_len, truncation=True, padding=False)
        labels = [t if t != self.tok.pad_token_id else -100 for t in tgt["input_ids"]]
        return {
            "input_ids": src["input_ids"],
            "attention_mask": src["attention_mask"],
            "labels": labels,
        }


def collate(batch, pad_id):
    max_src = max(len(b["input_ids"]) for b in batch)
    max_tgt = max(len(b["labels"]) for b in batch)
    out = {"input_ids": [], "attention_mask": [], "labels": []}
    for b in batch:
        sp = max_src - len(b["input_ids"])
        tp = max_tgt - len(b["labels"])
        out["input_ids"].append(b["input_ids"] + [pad_id] * sp)
        out["attention_mask"].append(b["attention_mask"] + [0] * sp)
        out["labels"].append(b["labels"] + [-100] * tp)
    return {k: torch.tensor(v) for k, v in out.items()}


def evaluate(model, loader, tokenizer, device, beam_size, max_target_length, max_batches=50):
    model.eval()
    total_loss = 0.0
    exact_match = 0
    total = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            total_loss += out.loss.item()

            if i < max_batches:
                gen = model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    num_beams=beam_size,
                    max_length=max_target_length,
                )
                pred = tokenizer.batch_decode(gen, skip_special_tokens=True)
                labels = batch["labels"].clone()
                labels[labels == -100] = tokenizer.pad_token_id
                truth = tokenizer.batch_decode(labels, skip_special_tokens=True)
                for p, t in zip(pred, truth):
                    if p.strip() == t.strip():
                        exact_match += 1
                    total += 1

    return {
        "loss": total_loss / max(len(loader), 1),
        "exact_match": exact_match / total if total else 0.0,
        "eval_examples": total,
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    print(f"loading {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.add_tokens(["<e>", "</e>"], special_tokens=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    train_ds = CorrectorDataset(
        Path(args.data_dir) / "corrector_train.jsonl",
        tokenizer, args.max_source_length, args.max_target_length,
        exclude=args.exclude_error_types, limit=args.max_train_examples,
    )
    val_ds = CorrectorDataset(
        Path(args.data_dir) / "corrector_val.jsonl",
        tokenizer, args.max_source_length, args.max_target_length,
        exclude=args.exclude_error_types,
    )
    print(f"train={len(train_ds)} val={len(val_ds)} (excluded: {args.exclude_error_types})")

    pad_id = tokenizer.pad_token_id
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=lambda b: collate(b, pad_id))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=lambda b: collate(b, pad_id))

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = (len(train_loader) // args.grad_accum) * args.epochs
    sched = get_linear_schedule_with_warmup(optim, int(args.warmup_ratio * total_steps), total_steps)

    best_em = -1.0
    history = []

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        optim.zero_grad()

        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}")
        for step, batch in enumerate(pbar):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / args.grad_accum
            loss.backward()
            running += loss.item() * args.grad_accum

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                sched.step()
                optim.zero_grad()

            if step % 50 == 0:
                pbar.set_postfix({"loss": f"{running/(step+1):.4f}"})

        metrics = evaluate(model, val_loader, tokenizer, device, args.beam_size, args.max_target_length)
        metrics["epoch"] = epoch + 1
        metrics["train_loss"] = running / len(train_loader)
        history.append(metrics)
        print(f"  val: {json.dumps(metrics, indent=2)}")

        if metrics["exact_match"] > best_em:
            best_em = metrics["exact_match"]
            save_dir = out_dir / "best"
            save_dir.mkdir(parents=True, exist_ok=True)
            try:
                model.save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)
            except Exception as e:
                print(f"  save_pretrained failed ({type(e).__name__}), saving manually")
                torch.save(model.state_dict(), save_dir / "pytorch_model.bin")
                model.config.save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)
            print(f"  saved best (em={best_em:.4f})")

    with (out_dir / "history.json").open("w") as f:
        json.dump(history, f, indent=2)
    print(f"done. best em={best_em:.4f}. artifacts in {out_dir}")


if __name__ == "__main__":
    main()