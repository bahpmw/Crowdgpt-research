#!/usr/bin/env python3
"""Turn FineWeb-Edu parquet shards into flat uint16 token bins.

The scaling test needs real tokens, not ``torch.randint``: a random-token
stream has no learnable signal, so every loss curve measured on it is a
mechanism smoke test. This reads the parquet shards the downloader left in
``--src``, tokenises with DUST's fixed BPE-4096 tokenizer, and writes

    <out>/train.bin   uint16, documents separated by <|endoftext|> (id 0)
    <out>/val.bin     the last ``--val-tokens`` tokens, held out of training

Usage:
    hf download HuggingFaceFW/fineweb-edu sample/10BT/000_00000.parquet \
        --repo-type dataset --local-dir /tmp/v15data/fw
    python tools/prepare_tokens.py --src /tmp/v15data/fw --out /tmp/v15data/tok \
        --tokens 40000000 --val-tokens 2000000

The tokenizer is DUST's (qlabs-eng/dust, MIT). When ``--tokenizer`` does not
exist it is downloaded from a pinned commit and checked against a pinned
sha256, so two machines that run this script get the same token ids.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

TOKENIZER_URL = ("https://raw.githubusercontent.com/qlabs-eng/dust/"
                 "bed52420091c415a2f4e822ab2d88cfc7fc7bae3/tokenizer/tokenizer.json")
TOKENIZER_SHA256 = "a15168ae52f4a1b1c80b270a5e18a2ce3e038b5e3772a1ad3982044602da0ae7"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure_tokenizer(path: Path, *, url: str = TOKENIZER_URL,
                     sha256: str = TOKENIZER_SHA256) -> str:
    """Return the tokenizer's sha256, downloading the pinned file if missing."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_suffix(path.suffix + ".part")
        with urllib.request.urlopen(url, timeout=60) as response:
            part.write_bytes(response.read())
        got = _sha256(part)
        if got != sha256:
            part.unlink()
            raise SystemExit(f"tokenizer sha256 {got} != pinned {sha256}")
        os.replace(part, path)
        print(f"fetched tokenizer -> {path}", flush=True)
    got = _sha256(path)
    if got != sha256:
        print(f"warning: {path} sha256 {got} is not the pinned DUST tokenizer",
              file=sys.stderr)
    return got


def _shards(src: Path) -> list[Path]:
    files = sorted(src.rglob("*.parquet"))
    if not files:
        raise SystemExit(f"no parquet shards under {src}")
    return files


def _texts(shard: Path):
    import pyarrow.parquet as pq

    table = pq.ParquetFile(shard)
    for batch in table.iter_batches(batch_size=2048, columns=["text"]):
        for value in batch.column("text").to_pylist():
            if value:
                yield value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", default="/tmp/v15data/fw")
    parser.add_argument("--out", default="/tmp/v15data/tok")
    parser.add_argument("--tokenizer", default="/tmp/dust/tokenizer/tokenizer.json",
                        help="tokenizer.json; fetched from the pinned DUST commit "
                             "when the file does not exist")
    parser.add_argument("--tokens", type=int, default=40_000_000)
    parser.add_argument("--val-tokens", type=int, default=2_000_000)
    parser.add_argument("--min-doc-chars", type=int, default=200)
    args = parser.parse_args(argv)

    from tokenizers import Tokenizer

    tokenizer_sha = ensure_tokenizer(Path(args.tokenizer))
    tokenizer = Tokenizer.from_file(args.tokenizer)
    eot = tokenizer.token_to_id("<|endoftext|>")
    if eot is None:
        raise SystemExit("tokenizer has no <|endoftext|> id")
    vocab = tokenizer.get_vocab_size()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    needed = int(args.tokens) + int(args.val_tokens)
    train_path = out / "train.bin"
    val_path = out / "val.bin"

    written = 0
    docs = 0
    started = time.perf_counter()
    tmp_train = out / "train.bin.part"
    with open(tmp_train, "wb") as fh:
        for shard in _shards(Path(args.src)):
            texts = [t for t in _texts(shard) if len(t) >= args.min_doc_chars]
            for start in range(0, len(texts), 512):
                chunk = texts[start:start + 512]
                encodings = tokenizer.encode_batch(chunk)
                for encoding in encodings:
                    ids = np.asarray(encoding.ids, dtype=np.uint16)
                    ids = np.concatenate([ids, np.array([eot], dtype=np.uint16)])
                    fh.write(ids.tobytes())
                    written += int(ids.size)
                    docs += 1
                    if written >= needed:
                        break
                if written >= needed:
                    break
                if written and docs % 2000 == 0:
                    rate = written / max(1e-9, time.perf_counter() - started)
                    print(f"{written/1e6:.1f}M tokens, {docs} docs, {rate/1e3:.0f}k tok/s",
                          flush=True)
            if written >= needed:
                break
    if written < needed:
        print(f"warning: only {written} tokens available (< {needed})", file=sys.stderr)
    os.replace(tmp_train, train_path)

    # Split the tail off as held-out data: whole documents are not tracked, so
    # split at a token boundary and accept the tiny chance of a crossed document.
    all_tokens = np.fromfile(train_path, dtype=np.uint16)
    val = all_tokens[-int(args.val_tokens):]
    train = all_tokens[:-int(args.val_tokens)]
    val.tofile(val_path)
    train.tofile(train_path)

    meta = {
        "schema": "crowd-v15-tokens-1",
        "tokenizer": args.tokenizer,
        "tokenizerSha256": tokenizer_sha,
        "vocabSize": int(vocab),
        "eot": int(eot),
        "dtype": "uint16",
        "trainTokens": int(train.size),
        "valTokens": int(val.size),
        "documents": docs,
        "source": str(args.src),
        "seconds": round(time.perf_counter() - started, 1),
    }
    import json
    (out / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(json.dumps(meta, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
