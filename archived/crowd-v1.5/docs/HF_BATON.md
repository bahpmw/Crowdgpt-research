# Handing the checkpoint around on the Hub

There is no coordinator to deploy for the first convergence runs. A Hugging
Face repo holds the baton — one weight file plus a round counter — and trainers
take turns: pull, train locally, push with the round incremented.

```
repo: peva3/Crowd-v1.5-dust    preset: 300m (306,222,080 params)
weights.safetensors            float32, 1.22 GB
status.json                    round, baseSha256, history
manifest.json                  names, sizes, shapes, baseSha256
seed.json                      preset, params, genesis digest
lock.json                      advisory 24 h lease
```

## Why this works with no server

The only thing a coordinator adds for a single trusted group is *ordering*. Here
ordering is the round number and the guard that refuses a push whose round is
not exactly one ahead of the remote. A stale trainer cannot silently win because
its push fails and it must pull again.

The digest is the second half: `baseSha256` is sha256 over sorted parameter names
plus the float32 values **rounded to bfloat16**, so it does not move when a peer
re-saves in another dtype, and it does move when weights change. Every pull and
push recomputes it.

## Token

Create a **fine-grained** token with `Repositories: write` limited to this one
repo, and pass it by environment variable (`HF_TOKEN`). Never commit it. A leaked
token is only able to overwrite this checkpoint, which is recoverable from git
history; a leaked *account* token would not be.

## Commands

```
python tools/hf_baton.py pull   --repo peva3/Crowd-v1.5-dust --dir state
python tools/hf_baton.py verify --dir state
python tools/hf_baton.py train  --repo peva3/Crowd-v1.5-dust --dir state \
    --data /path/to/tok --device cuda --steps 200 --batch 2 --seq-len 512 \
    --population 32 --draw-chunk 4 --mode all --push
```

`init` builds a fresh seed checkpoint locally (used once, to publish 300M);
`push --force` exists only for re-seeding, and tags the old revision first.

## Measured

300M, one 4060 Ti (16 GB): `--batch 2 --seq-len 512 --population 32
--draw-chunk 4 --mode all` → 57.5 s/step for 1,024 evidence tokens (17.9
tokens/s), 8.46 GB peak VRAM. That is the number to beat with parallelism: each
peer adds whole steps, and no all-reduce or gradient buffer is involved.

Honest limit: at ~18 evidence tokens/s a single 16 GB card cannot show
convergence at 300M in one sitting (306M params × 20 tokens/param ≈ 6.1B
tokens). The point of the baton is that N peers multiply that linearly.
