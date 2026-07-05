#!/usr/bin/env python3
"""
Convierte database.json (1.2 GB) en dos ficheros más ligeros:
  - embeddings.bin  → Int8Array binario cuantizado (138K × 384 × 1 byte ≈ 53 MB)
  - metadata.json   → texto, entidades, filename (sin embeddings) ≈ 30-50 MB
"""

import json
import sys
import time
import numpy as np

INPUT  = "database.json"
OUTPUT_BIN  = "embeddings.bin"
OUTPUT_META = "metadata.json"

print(f"📂 Leyendo {INPUT} ...", flush=True)
t0 = time.time()

with open(INPUT, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"✅ {len(data):,} chunks cargados en {time.time()-t0:.1f}s", flush=True)

# Validar que todos los chunks tienen embedding
dims = set(len(d["embedding"]) for d in data)
if len(dims) != 1:
    print(f"⚠️  Dimensiones inconsistentes: {dims}", file=sys.stderr)
    sys.exit(1)

dim = dims.pop()
print(f"📐 Dimensión de embeddings: {dim}", flush=True)

# --- 1. Guardar embeddings binarios ---
print(f"💾 Escribiendo {OUTPUT_BIN} ...", flush=True)
t1 = time.time()
# Cuantizamos de Float32 a Int8 (multiplicando por 127) para reducir el tamaño al 25%
emb_float = np.array([d["embedding"] for d in data], dtype=np.float32)
emb_int8 = np.clip(emb_float * 127, -127, 127).astype(np.int8)
emb_int8.tofile(OUTPUT_BIN)
bin_mb = emb_int8.nbytes / 1024 / 1024
print(f"✅ {OUTPUT_BIN} → {bin_mb:.1f} MB  ({time.time()-t1:.1f}s)", flush=True)

# --- 2. Guardar metadatos sin embeddings ---
print(f"💾 Escribiendo {OUTPUT_META} ...", flush=True)
t2 = time.time()
meta = [
    {k: v for k, v in d.items() if k != "embedding"}
    for d in data
]
with open(OUTPUT_META, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, separators=(",", ":"))
import os
meta_mb = os.path.getsize(OUTPUT_META) / 1024 / 1024
print(f"✅ {OUTPUT_META} → {meta_mb:.1f} MB  ({time.time()-t2:.1f}s)", flush=True)

# --- Resumen ---
print()
print("=" * 50)
print(f"  Chunks     : {len(data):,}")
print(f"  Dimensión  : {dim}")
print(f"  embeddings.bin : {bin_mb:.1f} MB")
print(f"  metadata.json  : {meta_mb:.1f} MB")
print(f"  Total nuevo    : {bin_mb + meta_mb:.1f} MB  (vs {1286:.0f} MB antes)")
print(f"  Reducción      : {(1 - (bin_mb + meta_mb) / 1286) * 100:.0f}%")
print("=" * 50)
print(f"⏱️  Tiempo total: {time.time()-t0:.1f}s")
print("🎉 ¡Conversión completada! Ahora actualiza app.js.")
