"""thin wrapper around ssl_trainer.train() that survives the missing-nvcc
crash on save.

context: model.encoder.save_pretrained() routes through transformers →
accelerate → deepspeed.unwrap_model, which probes $CUDA_HOME/bin/nvcc.
nvcc is not present in the singularity image, so the call raises
FileNotFoundError mid-training. this wrapper patches BertModel.save_pretrained
to catch that specific error and fall back to manual torch.save +
config.save_pretrained (the same pattern src/seq2seq.py already uses for
its corrector saves). the teammate's ssl_trainer.py stays unmodified.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import torch
import transformers


_orig_save = transformers.BertModel.save_pretrained


def _safe_save(self, save_dir, *args, **kwargs):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    try:
        return _orig_save(self, save_dir, *args, **kwargs)
    except FileNotFoundError as e:
        if "nvcc" in str(e):
            print("  save_pretrained hit deepspeed nvcc probe; saving manually")
            torch.save(self.state_dict(), save_dir / "pytorch_model.bin")
            self.config.save_pretrained(save_dir)
            return
        raise


transformers.BertModel.save_pretrained = _safe_save


from ssl_trainer import train

train()
