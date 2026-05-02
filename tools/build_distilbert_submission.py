"""Build a DistilBERT submission package.

Outputs:
  submission/model.py  — wrapper class with embedded tokenizer (base64-gzip-json bundle)
  submission/model.pt  — fp16 state dict (~134 MB)

The submission requires `transformers` on the backend. If the backend env
matches the documented spec (numpy/pandas/torch/torchvision/sklearn/opencv),
this submission will fail with ImportError on `from transformers import ...`.
The error message will tell us, and we pivot.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import io
import json
import sys
from pathlib import Path

import torch


MODEL_PY_TEMPLATE = '''"""DistilBERT-based news source classifier.

The tokenizer files (vocab.txt, tokenizer.json, configs) are gzipped and
base64-encoded inline; on first instantiation we extract them to a tmpdir
and load via the standard HuggingFace from_pretrained APIs.

Backend env requirement: transformers must be installed alongside torch.
"""

import base64
import gzip
import json
import os
import tempfile

import torch
from torch import nn

# Bundled tokenizer + model config files (gzipped+base64 of a JSON manifest)
_TOK_BUNDLE_B64 = """{tok_payload}"""


def _extract_tokenizer_dir():
    raw = base64.b64decode(_TOK_BUNDLE_B64.strip())
    manifest = json.loads(gzip.decompress(raw))
    d = tempfile.mkdtemp(prefix="distilbert_tok_")
    for fname, b64 in manifest.items():
        with open(os.path.join(d, fname), "wb") as f:
            f.write(base64.b64decode(b64))
    return d


_LABELS = {{0: "NBC", 1: "FoxNews"}}


class Model(nn.Module):
    """Wrapper around DistilBertForSequenceClassification.

    The backend will:
        m = Model()                  # __init__ here
        m.load_state_dict(state)     # backend loads model.pt
        m.predict(batch)             # called for inference
    """

    def __init__(self):
        super().__init__()
        # Lazy imports so module-import fails fast if transformers is missing
        from transformers import (
            DistilBertConfig,
            DistilBertForSequenceClassification,
            DistilBertTokenizerFast,
        )
        d = _extract_tokenizer_dir()
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(d)
        # Build architecture from config only — weights come from backend's load_state_dict()
        config = DistilBertConfig.from_pretrained(d)
        config.num_labels = 2
        config.id2label = {{0: "NBC", 1: "FoxNews"}}
        config.label2id = {{"NBC": 0, "FoxNews": 1}}
        self.bert = DistilBertForSequenceClassification(config)
        self.bert.eval()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert.to(self._device)

    def load_state_dict(self, state_dict, strict=False):
        # Backend hands us the saved state dict; route into the wrapped HF model.
        try:
            return self.bert.load_state_dict(state_dict, strict=strict)
        except Exception:
            # Tolerate prefix differences (e.g. "bert." or "distilbert.")
            stripped = {{}}
            for k, v in state_dict.items():
                nk = k
                for prefix in ("bert.", "distilbert.", "module."):
                    if nk.startswith(prefix):
                        nk = nk[len(prefix):]
                        break
                stripped[nk] = v
            return self.bert.load_state_dict(stripped, strict=False)

    @torch.no_grad()
    def predict(self, batch):
        if hasattr(batch, "tolist"):
            batch = batch.tolist()
        if isinstance(batch, str):
            batch = [batch]
        else:
            batch = [str(x) for x in batch]
        if not batch:
            return []
        # Tokenize once, run in chunks of 64 to bound memory
        results = []
        BS = 64
        self.bert.eval()
        for i in range(0, len(batch), BS):
            chunk = batch[i:i+BS]
            enc = self.tokenizer(chunk, padding=True, truncation=True, max_length=128, return_tensors="pt")
            enc = {{k: v.to(self._device) for k, v in enc.items()}}
            logits = self.bert(**enc).logits
            preds = logits.argmax(dim=-1).detach().cpu().tolist()
            results.extend(_LABELS[int(p)] for p in preds)
        return results

    def forward(self, batch):
        return self.predict(batch)


def get_model():
    return Model()
'''


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("models/distilbert_final"),
        help="dir with model.safetensors + tokenizer files + config.json",
    )
    p.add_argument(
        "--model-py-out",
        type=Path,
        default=Path("submission/model.py"),
    )
    p.add_argument(
        "--model-pt-out",
        type=Path,
        default=Path("submission/model.pt"),
    )
    p.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="save state dict in fp16 (halves file size)",
    )
    args = p.parse_args()

    # 1. Bundle tokenizer + config
    tok_files = {}
    for fname in ("tokenizer.json", "vocab.txt", "tokenizer_config.json",
                   "special_tokens_map.json", "config.json"):
        path = args.checkpoint_dir / fname
        if not path.exists():
            sys.exit(f"missing required file: {path}")
        tok_files[fname] = path.read_bytes()

    manifest = {k: base64.b64encode(v).decode("ascii") for k, v in tok_files.items()}
    payload = base64.b64encode(
        gzip.compress(json.dumps(manifest).encode("utf-8"), compresslevel=9)
    ).decode("ascii")
    print(f"tokenizer bundle: {len(payload):,} chars (~{len(payload)/1024:.0f} KB)")

    # 2. Write model.py
    args.model_py_out.parent.mkdir(parents=True, exist_ok=True)
    args.model_py_out.write_text(MODEL_PY_TEMPLATE.format(tok_payload=payload))
    print(f"wrote {args.model_py_out} ({args.model_py_out.stat().st_size:,} bytes)")

    # 3. Convert safetensors → state dict
    import safetensors.torch as st
    sd = st.load_file(str(args.checkpoint_dir / "model.safetensors"))
    if args.fp16:
        sd = {k: v.half() if v.dtype == torch.float32 else v for k, v in sd.items()}

    torch.save(sd, args.model_pt_out)
    print(f"wrote {args.model_pt_out} ({args.model_pt_out.stat().st_size/1e6:.1f} MB, fp16={args.fp16})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
