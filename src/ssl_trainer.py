"""
Self-Supervised Learning with Denoising Autoencoder (DAE)

This is TRUE Self-Supervised Learning:
- Input: Unlabeled raw text (no annotations needed)
- Process: Add corruption → Learn to reconstruct
- Output: Pre-trained representations useful for downstream tasks
- Benefit: Detector and Seq2Seq benefit from SSL pre-training

The DAE learns:
1. Robust text representations
2. How to handle corrupted/noisy input
3. Similarity between corrupted and clean text
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent))

from utils import set_seed
from ssl_corruption import TextCorruptor, edit_distance


class UnlabeledTextDataset(Dataset):
    """Dataset for unlabeled text (used for SSL pre-training)"""
    
    def __init__(self, text_file, tokenizer, max_length=192, limit=-1,
                 corruption_intensity=0.5, corruption_type='mixed',
                 min_edit_distance=2):
        """
        Args:
            text_file: Path to file with one text per line
            tokenizer: Hugging Face tokenizer
            max_length: Max sequence length
            limit: Max examples to use (-1 for all)
            corruption_intensity: Intensity for curriculum scheduling (0.0-1.0)
            corruption_type: 'light', 'medium', 'heavy', 'social', 'mixed'
            min_edit_distance: Skip examples where corruption changes fewer chars (difficulty filter)
        """
        self.texts = []
        
        # Support both plain text and jsonl format
        with open(text_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                
                # Try to parse as JSON, fall back to plain text
                try:
                    row = json.loads(line)
                    text = row.get('text', row.get('content', line))
                except:
                    text = line
                
                self.texts.append(text)
                
                if limit > 0 and len(self.texts) >= limit:
                    break
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.corruptor = TextCorruptor()
        self.corruption_intensity = corruption_intensity
        self.corruption_type = corruption_type
        self.min_edit_distance = min_edit_distance
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        clean_text = self.texts[idx]

        # Difficulty filter: retry up to 5 times to get a non-trivial corruption
        for _ in range(5):
            corrupted_text = self.corruptor.apply_corruption(
                clean_text,
                corruption_type=self.corruption_type,
                intensity=self.corruption_intensity,
            )
            if edit_distance(clean_text, corrupted_text) >= self.min_edit_distance:
                break
        
        # Tokenize clean and corrupted versions
        clean_enc = self.tokenizer(
            clean_text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        corrupted_enc = self.tokenizer(
            corrupted_text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': corrupted_enc['input_ids'].squeeze(0),
            'attention_mask': corrupted_enc['attention_mask'].squeeze(0),
            'labels': clean_enc['input_ids'].squeeze(0),
        }


class DenoisingAutoencoder(nn.Module):
    """
    Denoising Autoencoder for SSL
    
    Architecture:
    - Encoder: Pretrained transformer (e.g., RoBERT)
    - Decoder: Transformer decoder that reconstructs tokens
    - Loss: Reconstruction loss (only on non-padding tokens)
    """
    
    def __init__(self, model_name: str, hidden_size: int = 1024, vocab_size: int = 50265):
        super().__init__()
        self.model_name = model_name
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        # Encoder: Pretrained transformer
        self.encoder = AutoModel.from_pretrained(model_name)
        self.encoder_dim = self.encoder.config.hidden_size
        
        # Projection to vocab size for reconstruction
        self.reconstruction_head = nn.Linear(self.encoder_dim, vocab_size)
    
    def forward(self, input_ids, attention_mask, labels=None):
        """
        Args:
            input_ids: Corrupted token ids (batch_size, seq_len)
            attention_mask: Attention mask for input
            labels: Clean token ids for reconstruction loss (optional)
        
        Returns:
            Dictionary with logits, loss (if labels provided)
        """
        # Encode corrupted input
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Get last hidden state (batch_size, seq_len, hidden_dim)
        hidden_states = encoder_output.last_hidden_state
        
        # Project to vocabulary size for reconstruction
        logits = self.reconstruction_head(hidden_states)  # (batch_size, seq_len, vocab_size)
        
        outputs = {'logits': logits}
        
        if labels is not None:
            # Reconstruction loss: predict clean tokens from corrupted input
            # Only compute loss on non-padding positions
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            
            # Flatten for loss computation
            logits_flat = logits.view(-1, self.vocab_size)
            labels_flat = labels.view(-1)
            attention_flat = attention_mask.view(-1)
            
            # Compute loss only on attended tokens
            loss = loss_fct(logits_flat, labels_flat)
            loss = (loss * attention_flat).sum() / (attention_flat.sum() + 1e-8)
            
            outputs['loss'] = loss
        
        return outputs


def parse_args():
    p = argparse.ArgumentParser(description="SSL Pre-training with Denoising Autoencoder")
    p.add_argument("--unlabeled_data", type=str, required=True, help="Path to unlabeled text file")
    p.add_argument("--out_dir", type=str, default="results/ssl_dae", help="Output directory")
    p.add_argument("--model_name", type=str, default="readerbench/RoBERT-large", help="Base model")
    p.add_argument("--max_length", type=int, default=192)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--max_examples", type=int, default=-1, help="Max examples to use (-1 for all)")
    p.add_argument("--save_every", type=int, default=500, help="Save checkpoint every N steps")
    p.add_argument("--log_every", type=int, default=100, help="Log every N steps")
    # Curriculum scheduling
    p.add_argument("--curriculum", action="store_true", help="Enable noise curriculum (intensity increases per epoch)")
    p.add_argument("--min_intensity", type=float, default=0.3, help="Starting corruption intensity (curriculum)")
    p.add_argument("--max_intensity", type=float, default=0.8, help="Final corruption intensity (curriculum)")
    p.add_argument("--min_edit_distance", type=int, default=2, help="Min edit distance for difficulty filter")
    return p.parse_args()


def train():
    args = parse_args()
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🔧 Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = DenoisingAutoencoder(args.model_name)
    model.to(device)
    
    print(f"📚 Loading unlabeled data: {args.unlabeled_data}")
    # Initial dataset - intensity will be updated per epoch if curriculum is on
    initial_intensity = args.min_intensity if args.curriculum else 0.6

    def make_dataset(intensity, corruption_type='mixed'):
        return UnlabeledTextDataset(
            args.unlabeled_data,
            tokenizer,
            max_length=args.max_length,
            limit=args.max_examples,
            corruption_intensity=intensity,
            corruption_type=corruption_type,
            min_edit_distance=args.min_edit_distance,
        )

    dataset = make_dataset(initial_intensity)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"📊 Dataset size: {len(dataset)} examples")
    print(f"💾 Saving to: {out_dir}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    total_steps = len(dataloader) * args.epochs // args.grad_accum
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    model.train()
    global_step = 0
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Curriculum: increase intensity and difficulty per epoch
        if args.curriculum and args.epochs > 1:
            progress = epoch / (args.epochs - 1)  # 0.0 → 1.0
            intensity = args.min_intensity + progress * (args.max_intensity - args.min_intensity)
            # Epoch schedule: light → medium → heavy/mixed
            if progress < 0.33:
                corruption_type = 'light'
            elif progress < 0.66:
                corruption_type = 'medium'
            else:
                corruption_type = 'mixed'
            print(f"📈 Curriculum epoch {epoch+1}: intensity={intensity:.2f}, type={corruption_type}")
            dataset = make_dataset(intensity, corruption_type)
            dataloader = DataLoader(
                dataset, batch_size=args.batch_size, shuffle=True,
                num_workers=args.num_workers, pin_memory=True
            )
        epoch_loss = 0.0
        batch_count = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}", unit="batch")
        
        for batch_idx, batch in enumerate(pbar):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            loss = outputs['loss']
            
            # Backward pass
            loss = loss / args.grad_accum
            loss.backward()
            
            if (batch_idx + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
            
            epoch_loss += loss.item() * args.grad_accum
            batch_count += 1
            
            # Logging
            if global_step % args.log_every == 0:
                avg_loss = epoch_loss / max(batch_count, 1)
                pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
            
            # Save checkpoint
            if global_step % args.save_every == 0 and global_step > 0:
                checkpoint_dir = out_dir / f"checkpoint-{global_step}"
                checkpoint_dir.mkdir(exist_ok=True)
                model.encoder.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)
                print(f"💾 Saved checkpoint at step {global_step}")
        
        avg_epoch_loss = epoch_loss / batch_count
        print(f"✓ Epoch {epoch+1} - Avg Loss: {avg_epoch_loss:.4f}")
        
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_dir = out_dir / "best"
            best_dir.mkdir(exist_ok=True)
            model.encoder.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            print(f"🏆 Best model saved (loss={best_loss:.4f})")
    
    # Save final model
    final_dir = out_dir / "final"
    final_dir.mkdir(exist_ok=True)
    model.encoder.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"✅ Training complete! Final model saved to {final_dir}")
    
    # Save training info
    info = {
        'model_name': args.model_name,
        'best_loss': float(best_loss),
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'max_length': args.max_length,
        'dataset_size': len(dataset),
        'curriculum': args.curriculum,
        'min_intensity': args.min_intensity,
        'max_intensity': args.max_intensity,
        'min_edit_distance': args.min_edit_distance,
    }
    with open(out_dir / "training_info.json", 'w') as f:
        json.dump(info, f, indent=2)


if __name__ == '__main__':
    train()
