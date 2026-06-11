# Web RAG LLM (In-Browser GraphRAG)

Este proyecto es un motor de búsqueda semántica y de razonamiento que funciona **100% de forma local en el navegador del usuario**, garantizando cero costes de servidor y máxima privacidad.

## Arquitectura

El proyecto se divide en dos fases:
1. **Servidor (Python):** El archivo `indexer.py` lee un conjunto de documentos (`.txt`), los trocea, extrae sus entidades principales usando **GLiNER**, genera vectores de embeddings semánticos y lo empaqueta todo en un archivo estático `database.json`.
2. **Cliente (Navegador):** Los archivos `index.html` y `app.js` cargan el archivo estático. Utilizan **Transformers.js** (WASM) para vectorizar las búsquedas del usuario y **WebLLM** (Qwen 0.5B vía WebGPU) para leer los documentos relevantes y responder preguntas. Además, utiliza **D3.js** para dibujar un Grafo de Conocimiento (Knowledge Graph) dinámico con las entidades recuperadas.

---

## 🛠️ Cómo procesar nuevos documentos

Si quieres añadir nuevas transcripciones o documentos:
1. Coloca tus archivos `.txt` en la carpeta `../transcripts` relativa a este directorio.
2. Asegúrate de tener las dependencias de Python instaladas:
   ```bash
   pip install -r requirements.txt
   ```
3. Ejecuta el indexador:
   ```bash
   python indexer.py
   ```
4. Se generará o sobreescribirá el archivo `database.json`. Refresca tu web y estará lista.

---

## 🚀 Despliegue en GitHub Pages

Al ser 100% estático (no hay backend en vivo), esta carpeta se puede subir a GitHub Pages:
1. Inicializa git en esta carpeta: `git init`
2. Añade los archivos: `git add .`
3. Haz el commit: `git commit -m "Initial commit"`
4. Empújalo a un nuevo repositorio de GitHub y activa GitHub Pages en las opciones (*Settings > Pages > Deploy from a branch > main*).

---

## ⚠️ Solución de Problemas: Activar WebGPU en Linux

La inteligencia artificial que te responde (WebLLM) requiere **WebGPU** para usar la tarjeta gráfica de tu ordenador.
Sin embargo, **todos los navegadores en Linux tienen WebGPU desactivado por defecto**, sin importar lo potente que sea tu tarjeta gráfica.

Si al abrir la web ves un error de GPU, sigue estos pasos para activarlo en 3 clics:

### En Chrome / Chromium / Brave
1. Escribe esto en tu barra de direcciones: `chrome://flags/#enable-unsafe-webgpu`
2. Cambia la opción de *Default* a **Enabled**.
3. (Opcional pero recomendado en Linux): Busca también `Vulkan` (`chrome://flags/#enable-vulkan`) y ponlo en **Enabled**.
4. Haz clic en el botón inferior de **Relaunch** para reiniciar el navegador.

### En Firefox
1. Escribe en la barra de direcciones: `about:config`
2. Acepta el aviso de riesgo.
3. Busca la propiedad `dom.webgpu.enabled` y cámbiala a **true**.
4. Reinicia el navegador.
