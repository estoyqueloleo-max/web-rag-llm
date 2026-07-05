Me dejas en el README de web-rag estas configuraiconeS? lincamos en la web a la seccion del readme?

y si hacemos que los nodos sean lincables y relancen la busqueda? para ver como avanza?

ponemos en el readme todo el proceso?

y lo lanzamos con todo la carpeta .... que deberiamos mover dentro de la carpeta web... a ver como salen de tamaño los ficheros?

Y vamos preparando un github pages para mover la carpeta a un repo nuevo?

---

## 🗄️ Problema: `database.json` demasiado grande (1,2 GB)

El fichero actual contiene **138.936 chunks** de **1.325 vídeos** de YouTube.
Cada chunk almacena un embedding de 384 dimensiones como array JSON de floats, lo que hace el fichero completamente inusable en el navegador (minutos de descarga, varios GB de RAM al parsear).

Stats actuales:
- `database.json` → **1,2 GB**
- Chunks: 138.936 | Archivos fuente: 1.325 | Dimensión embedding: 384

### ✅ Opción A — Formato binario (ELEGIDA)
Separar embeddings y metadatos en dos ficheros:

1. **`embeddings.bin`** → todos los vectores como `Float32Array` binario puro  
   Tamaño estimado: `138.936 × 384 × 4 bytes ≈ 213 MB` (~150 MB con gzip HTTP)

2. **`metadata.json`** → solo `id`, `filename`, `text`, `entities` (sin embeddings)  
   Tamaño estimado: ~30–50 MB comprimido

**En `app.js`**: sustituir el `fetch('database.json')` por:
```js
// Cargar metadatos (texto, entidades)
const meta = await fetch('metadata.json').then(r => r.json());

// Cargar embeddings como buffer binario
const buf = await fetch('embeddings.bin').then(r => r.arrayBuffer());
const embeddings = new Float32Array(buf); // embeddings[i*384 .. (i+1)*384]
```
La búsqueda por coseno itera sobre slices del `Float32Array`, igual que ahora.

**Script de conversión** (`convert_db.py`) a crear:
```python
import json, struct, numpy as np

with open('database.json') as f:
    data = json.load(f)

# Guardar embeddings binarios
emb = np.array([d['embedding'] for d in data], dtype=np.float32)
emb.tofile('embeddings.bin')

# Guardar metadatos sin embeddings
meta = [{k: v for k, v in d.items() if k != 'embedding'} for d in data]
with open('metadata.json', 'w') as f:
    json.dump(meta, f)
```

**Ventaja**: mantiene el sistema 100% serverless y es compatible con GitHub Pages.

---

### Opción B — Servidor local mínimo (FastAPI/Go)
El backend hace la búsqueda vectorial con numpy/FAISS y devuelve solo los top-5.  
El frontend no carga nada pesado. Requiere proceso corriendo → no válido para GitHub Pages.

### Opción C — Reducir el dataset
Filtrar solo un subconjunto temático (ej. geopolítica 2025–2026).  
Solución parcial; no escala si se añaden más vídeos.

---

## 🔧 Tareas Pendientes a Futuro / Troubleshooting
- **Arreglar IPEX (`intel_extension_for_pytorch`) en Linux**:
  Al intentar usar la GPU Intel para inferencia o entrenamiento pesado en PyTorch, la librería (`libintel-ext-pt-cpu.so`) falla con el error de seguridad: 
  `cannot enable executable stack as shared object requires: Invalid argument`.
  **Solución**: Instalar la utilidad `execstack` y limpiar la bandera de seguridad de la librería ejecutando `sudo execstack -c /ruta/a/libintel-ext-pt-cpu.so`, o reinstalar una versión que sea compatible con las políticas del kernel.
## Sugerencias Pendientes de Implementar (Mejoras RAG)
- [ ] **Chunking Inteligente:** Cambiar la partición de 500 caracteres por un tokenizador o divisor semántico (ej. `RecursiveCharacterTextSplitter` de LangChain) para no cortar oraciones por la mitad.
- [ ] **Metadatos de Tiempo (Timestamps):** Indexar las transcripciones usando formato `.vtt` o `.srt` para que los chunks de la base de datos tengan un `minuto` asociado y se pueda navegar al video en ese momento exacto.
- [ ] **Añadir fuentes al Prompt del LLM:** Modificar el `context` en `app.js` para añadir el título del vídeo (`r.filename`) antes del texto de cada chunk, permitiendo a Qwen referenciar correctamente la fuente.
