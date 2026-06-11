import os
import json
import torch
try:
    import intel_extension_for_pytorch as ipex
except ImportError:
    pass

from sentence_transformers import SentenceTransformer
from gliner import GLiNER

TRANSCRIPTS_DIR = "transcripts"
OUTPUT_DB = "database.json"
LIMIT = None  # Procesar todos los archivos
CHUNK_SIZE = 500 # Caracteres por chunk
OVERLAP = 100

def chunk_text(text, chunk_size, overlap):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    return chunks

def main():
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        device = "xpu"
    else:
        device = "cpu"
        
    print(f"--- Arrancando indexador en el dispositivo: {device.upper()} ---")

    print("Cargando modelo de embeddings (paraphrase-multilingual-MiniLM-L12-v2)...")
    # Es vital usar el MISMO modelo aquí y en Javascript.
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', device=device)
    
    print("Cargando modelo GLiNER para extracción de entidades...")
    gliner_model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1").to(device)
    labels = [
        "Persona Histórica o Política", 
        "País o Región", 
        "Organización Institucional", 
        "Concepto Geopolítico o Económico", 
        "Evento o Conflicto"
    ]
    
    if not os.path.exists(TRANSCRIPTS_DIR):
        print(f"Error: La carpeta {TRANSCRIPTS_DIR} no existe.")
        return
        
    files = [f for f in os.listdir(TRANSCRIPTS_DIR) if f.endswith('.txt')]
    if LIMIT:
        files = files[:LIMIT]
        print(f"Limitando a {LIMIT} archivos por rapidez en este MVP.")
        
    database = []
    
    for filename in files:
        filepath = os.path.join(TRANSCRIPTS_DIR, filename)
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read().strip()
            
        if not text: continue
        
        # Generar chunks (fragmentos)
        chunks = chunk_text(text, CHUNK_SIZE, OVERLAP)
        
        for i, chunk in enumerate(chunks):
            # Calcular el vector (embedding)
            # Normalizamos para usar similitud del coseno correctamente
            emb = model.encode(chunk, normalize_embeddings=True).tolist()
            
            # Extraer entidades con GLiNER
            extracted = gliner_model.predict_entities(chunk, labels)
            entities = list(set([e["text"] for e in extracted])) # Eliminamos duplicados
            
            database.append({
                "id": f"{filename}_chunk_{i}",
                "filename": filename,
                "text": chunk,
                "embedding": emb,
                "entities": entities
            })
            
    print(f"Guardando {len(database)} fragmentos en {OUTPUT_DB}...")
    with open(OUTPUT_DB, 'w', encoding='utf-8') as f:
        json.dump(database, f, ensure_ascii=False)
        
    print("¡Base de datos generada con éxito!")

if __name__ == "__main__":
    main()
