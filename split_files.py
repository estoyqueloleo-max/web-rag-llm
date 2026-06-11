#!/usr/bin/env python3
"""
Divide embeddings.bin y metadata.json en partes < 95 MB
para poder subirlos a GitHub (límite 100 MB por fichero).

Genera:
  embeddings.bin.part0, .part1, .part2  (~68 MB cada una)
  metadata.part0.json, metadata.part1.json (~54 MB cada una)

El app.js los recombina en el navegador.
"""
import json, math, os, struct, time
import numpy as np

DIM       = 384
PART_MB   = 90          # Máximo MB por parte (con margen)
BIN_IN    = "embeddings.bin"
META_IN   = "metadata.json"

# ─── Partir embeddings.bin ──────────────────────────────────────────────────
print(f"📂 Leyendo {BIN_IN}…")
emb = np.fromfile(BIN_IN, dtype=np.float32).reshape(-1, DIM)
n_chunks = emb.shape[0]
bytes_per_chunk = DIM * 4
chunks_per_part = math.floor((PART_MB * 1024 * 1024) / bytes_per_chunk)
n_parts = math.ceil(n_chunks / chunks_per_part)

print(f"  {n_chunks:,} chunks → {n_parts} partes de ≤{chunks_per_part:,} chunks cada una")
for i in range(n_parts):
    start = i * chunks_per_part
    end   = min(start + chunks_per_part, n_chunks)
    part  = emb[start:end]
    fname = f"embeddings.bin.part{i}"
    part.tofile(fname)
    mb = os.path.getsize(fname) / 1024 / 1024
    print(f"  ✅ {fname} → {mb:.1f} MB  ({end-start:,} chunks)")

# ─── Partir metadata.json ────────────────────────────────────────────────────
print(f"\n📂 Leyendo {META_IN}…")
t0 = time.time()
with open(META_IN, encoding="utf-8") as f:
    meta = json.load(f)
print(f"  {len(meta):,} registros cargados en {time.time()-t0:.1f}s")

# Estimar chunks_per_part a partir del tamaño real del fichero
meta_bytes = os.path.getsize(META_IN)
meta_parts = math.ceil(meta_bytes / (PART_MB * 1024 * 1024))
recs_per_part = math.ceil(len(meta) / meta_parts)
print(f"  → {meta_parts} partes de ≤{recs_per_part:,} registros cada una")

for i in range(meta_parts):
    start = i * recs_per_part
    end   = min(start + recs_per_part, len(meta))
    fname = f"metadata.part{i}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(meta[start:end], f, ensure_ascii=False, separators=(",", ":"))
    mb = os.path.getsize(fname) / 1024 / 1024
    print(f"  ✅ {fname} → {mb:.1f} MB  ({end-start:,} registros)")

# ─── Resumen ─────────────────────────────────────────────────────────────────
print("\n" + "="*55)
total_emb  = sum(os.path.getsize(f"embeddings.bin.part{i}") for i in range(n_parts))
total_meta = sum(os.path.getsize(f"metadata.part{i}.json") for i in range(meta_parts))
print(f"  embeddings partes : {n_parts}  ({total_emb/1024/1024:.1f} MB total)")
print(f"  metadata partes   : {meta_parts}  ({total_meta/1024/1024:.1f} MB total)")
print(f"  Partes listas para Git (<95 MB cada una)")
print("="*55)
print("🎉 Ahora actualiza NUM_EMB_PARTS y NUM_META_PARTS en app.js si cambian.")
print(f"   NUM_EMB_PARTS  = {n_parts}")
print(f"   NUM_META_PARTS = {meta_parts}")
