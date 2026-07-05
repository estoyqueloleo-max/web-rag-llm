import { pipeline } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.1';
import * as webllm from "https://esm.run/@mlc-ai/web-llm";

let extractor;
let metadata = [];      // [{id, filename, text, entities}, ...]
let embeddings = null;  // Float32Array, dimensión DIM por chunk
const DIM = 384;
const NUM_EMB_PARTS  = 1;   // partes de embeddings.bin.part0..N-1
const NUM_META_PARTS = 2;   // partes de metadata.part0.json..N-1
let engine;

// Cache API para evitar descargar la BD cada vez
async function fetchWithCache(url) {
    const cache = await caches.open('rag-db-cache-v1');
    let response = await cache.match(url);
    if (!response) {
        response = await fetch(url);
        if (response.ok) {
            cache.put(url, response.clone());
        } else {
            throw new Error(`Recurso no encontrado: ${url}`);
        }
    }
    return response;
}

// Similitud coseno entre un array de consulta y un slice del buffer de embeddings
function cosineSimilarity(queryVec, allEmbeddings, idx) {
    let dot = 0, normA = 0, normB = 0;
    const offset = idx * DIM;
    for (let i = 0; i < DIM; i++) {
        const a = queryVec[i];
        const b = allEmbeddings[offset + i];
        dot   += a * b;
        normA += a * a;
        normB += b * b;
    }
    if (normA === 0 || normB === 0) return 0;
    return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

async function init() {
    const statusEl = document.getElementById('status');
    
    try {
        // 1. Cargar el mismo modelo usado en Python
        statusEl.textContent = "Cargando modelo de embeddings (WASM)...";
        extractor = await pipeline('feature-extraction', 'Xenova/paraphrase-multilingual-MiniLM-L12-v2');
        
        // 2a. Cargar metadatos en paralelo (con caché persistente)
        statusEl.textContent = "Cargando metadatos (desde caché si existe)...";
        const metaPromises = Array.from({ length: NUM_META_PARTS }, (_, i) =>
            fetchWithCache(`metadata.part${i}.json`)
                .then(r => r.json())
        );
        const metaParts = await Promise.all(metaPromises);
        metadata = metaParts.flat();

        // 2b. Cargar embeddings binarios Int8 en paralelo (con caché persistente)
        statusEl.textContent = "Cargando embeddings (Int8) binarios...";
        const embPromises = Array.from({ length: NUM_EMB_PARTS }, (_, i) =>
            fetchWithCache(`embeddings.bin.part${i}`)
                .then(r => r.arrayBuffer())
        );
        const embParts = await Promise.all(embPromises);
        // Concatenar todos los ArrayBuffers en un solo Int8Array
        const totalBytes = embParts.reduce((s, b) => s + b.byteLength, 0);
        const combined   = new Uint8Array(totalBytes);
        let offset = 0;
        for (const buf of embParts) { combined.set(new Uint8Array(buf), offset); offset += buf.byteLength; }
        embeddings = new Int8Array(combined.buffer);
        
        // 3. Cargar WebLLM (Qwen) con detección de capacidades WebGPU
        statusEl.textContent = "Detectando capacidades de la tarjeta gráfica...";
        
        let modelId = "Qwen2.5-0.5B-Instruct-q4f32_1-MLC"; // Fallback seguro para Android/GPUs antiguas
        if (navigator.gpu) {
            const adapter = await navigator.gpu.requestAdapter();
            if (adapter && adapter.features.has("shader-f16")) {
                modelId = "Qwen2.5-0.5B-Instruct-q4f16_1-MLC"; // Versión que consume menos RAM
                console.log("Soporte f16 detectado. Usando modelo optimizado:", modelId);
            } else {
                console.log("Sin soporte f16. Usando modelo de compatibilidad:", modelId);
            }
        }
        
        statusEl.textContent = `Cargando WebLLM (${modelId})... Esto puede tardar varios minutos y descargar ~350MB la primera vez.`;
        
        const initProgressCallback = (initProgress) => {
            statusEl.textContent = `Descargando LLM: ${initProgress.text}`;
        }
        
        engine = await webllm.CreateMLCEngine(
            modelId, 
            { initProgressCallback }
        );
        
        // 4. Listo para buscar
        statusEl.textContent = `¡Listo! Base de datos cargada con ${metadata.length} fragmentos de transcripciones.`;
        document.getElementById('searchInput').disabled = false;
        document.getElementById('searchBtn').disabled = false;
        
    } catch (error) {
        if (error.message.includes("Unable to find a compatible GPU") || error.message.includes("WebGPU")) {
            statusEl.innerHTML = `<span style="color: #d9534f;"><b>Error de WebGPU:</b> Tu navegador tiene WebGPU desactivado por defecto.</span><br>
            Consulta la sección <a href="README.md" target="_blank" style="color: #0066cc; font-weight: bold;">Solución de problemas en el README</a> para ver cómo activarlo fácilmente en Linux.`;
        } else {
            statusEl.textContent = `Error: ${error.message}`;
        }
        console.error(error);
    }
}

// Resalta las entidades dentro de un texto escapando HTML y envolviendo matches en <mark>
function highlightEntities(text, entities) {
    if (!entities || entities.length === 0) {
        return escapeHtml(text);
    }
    // Ordenar por longitud descendente para evitar solapamientos parciales
    const sorted = [...entities].sort((a, b) => b.length - a.length);
    // Escapar el texto base primero
    let result = escapeHtml(text);
    // Reemplazar cada entidad por su versión marcada (case-insensitive)
    sorted.forEach(entity => {
        const escaped = escapeHtml(entity);
        const regex = new RegExp(`(${escaped.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
        result = result.replace(regex, '<mark class="entity-highlight">$1</mark>');
    });
    return result;
}

function escapeHtml(str) {
    return str
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

// Dibujar grafo de fuerzas D3.js con zoom/pan y nodos arrastables
function drawGraph(topResults) {
    const svgEl = document.getElementById('graph');
    svgEl.style.display = 'block';
    svgEl.innerHTML = ''; // Limpiar SVG anterior

    // Dimensiones: 100% ancho, altura fija mayor
    const width  = svgEl.clientWidth  || 900;
    const height = 500;
    svgEl.setAttribute('height', height);
    svgEl.setAttribute('viewBox', `0 0 ${width} ${height}`);

    const nodesMap = new Map();
    const linksMap = new Map();

    // Asignar grupo por fuente (hasta 5 colores diferentes)
    const sourceColors = ['#4f8ef7','#f76b4f','#4fcf70','#f7c94f','#c44ff7'];
    topResults.forEach((res, resIdx) => {
        const ents = res.entities || [];
        ents.forEach(e => {
            if (!nodesMap.has(e)) nodesMap.set(e, { id: e, group: resIdx % sourceColors.length });
        });
        for (let i = 0; i < ents.length; i++) {
            for (let j = i + 1; j < ents.length; j++) {
                const linkId = [ents[i], ents[j]].sort().join('|||');
                if (!linksMap.has(linkId)) linksMap.set(linkId, { source: ents[i], target: ents[j], value: 1 });
                else linksMap.get(linkId).value++;
            }
        }
    });

    const nodes = Array.from(nodesMap.values());
    const links = Array.from(linksMap.values());

    if (nodes.length === 0) { svgEl.style.display = 'none'; return; }

    const svg = d3.select('#graph');

    // Capa raíz sobre la que se aplica el zoom
    const root = svg.append('g').attr('class', 'zoom-root');

    // --- Zoom / Pan ---
    const zoom = d3.zoom()
        .scaleExtent([0.2, 6])
        .on('zoom', (event) => root.attr('transform', event.transform));
    svg.call(zoom);
    // Doble clic resetea el zoom
    svg.on('dblclick.zoom', () => svg.transition().duration(500).call(zoom.transform, d3.zoomIdentity));

    // --- Simulación de fuerzas ---
    const simulation = d3.forceSimulation(nodes)
        .force('link',   d3.forceLink(links).id(d => d.id).distance(120))
        .force('charge', d3.forceManyBody().strength(-350))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collision', d3.forceCollide(30));

    // --- Aristas ---
    const link = root.append('g').attr('class', 'links')
        .selectAll('line')
        .data(links)
        .join('line')
        .attr('class', 'link')
        .attr('stroke-width', d => Math.max(1, Math.sqrt(d.value) * 1.5))
        .attr('stroke-opacity', 0.6);

    // --- Nodos (grupo g para círculo + texto) ---
    const node = root.append('g').attr('class', 'nodes')
        .selectAll('g')
        .data(nodes)
        .join('g')
        .style('cursor', 'pointer');

    // Drag: mueve solo el nodo, no confunde con el zoom
    node.call(
        d3.drag()
            .filter(event => !event.button && !event.ctrlKey)
            .on('start', (event, d) => {
                if (!event.active) simulation.alphaTarget(0.3).restart();
                d.fx = d.x; d.fy = d.y;
            })
            .on('drag',  (event, d) => { d.fx = event.x; d.fy = event.y; })
            .on('end',   (event, d) => {
                if (!event.active) simulation.alphaTarget(0);
                d.fx = null; d.fy = null;
            })
    );

    // Círculo con color por grupo
    node.append('circle')
        .attr('r', 14)
        .attr('fill', d => sourceColors[d.group])
        .attr('stroke', '#fff')
        .attr('stroke-width', 2)
        .on('click', (event, d) => {
            event.stopPropagation(); // no propagar al zoom
            document.getElementById('searchInput').value = d.id;
            search();
        });

    // Texto con fondo semitransparente para legibilidad
    node.append('text')
        .text(d => d.id)
        .attr('x', 18)
        .attr('y', 4)
        .style('font-size', '12px')
        .style('font-weight', '600')
        .style('paint-order', 'stroke')
        .style('stroke', 'rgba(0,0,0,0.55)')
        .style('stroke-width', '3px')
        .style('fill', '#fff');

    // Tooltip nativo con título
    node.append('title').text(d => `🔍 Buscar: "${d.id}"`);

    // Tick: actualizar posiciones
    simulation.on('tick', () => {
        link
            .attr('x1', d => d.source.x)
            .attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x)
            .attr('y2', d => d.target.y);
        node.attr('transform', d => `translate(${d.x},${d.y})`);
    });

    // Hint visual
    svg.append('text')
        .attr('x', 10).attr('y', height - 10)
        .style('font-size', '11px')
        .style('fill', '#aaa')
        .text('🖱 Scroll=zoom · Arrastra nodos · Doble clic=reset · Clic nodo=buscar');
}

async function search() {
    const query = document.getElementById('searchInput').value.trim();
    if (!query) return;
    
    const resultsEl = document.getElementById('results');
    const statusEl = document.getElementById('status');
    const llmEl = document.getElementById('llm-response');
    
    statusEl.textContent = "Vectorizando pregunta y buscando coincidencias...";
    resultsEl.innerHTML = "";
    llmEl.style.display = "none";
    document.getElementById('graph').style.display = "none";
    document.getElementById('searchBtn').disabled = true;
    
    try {
        // 1. Convertir la pregunta a un vector (embedding) localmente
        const output = await extractor(query, { pooling: 'mean', normalize: true });
        const queryEmbedding = Array.from(output.data);
        
        // 2. Calcular la similitud contra todos los chunks (sobre Float32Array binario)
        const queryLower = query.toLowerCase();
        const queryVec = new Float32Array(queryEmbedding);
        const scoredChunks = metadata.map((chunk, idx) => {
            let score = cosineSimilarity(queryVec, embeddings, idx);
            
            // Búsqueda híbrida: Boost si las palabras clave están en el texto
            const queryWords = queryLower.split(' ').filter(w => w.length > 3);
            let matchCount = 0;
            for (const w of queryWords) {
                if (chunk.text.toLowerCase().includes(w)) matchCount++;
            }
            if (queryWords.length > 0) {
                score += (matchCount / queryWords.length) * 0.2; // Hasta 0.2 extra si están todas las palabras
            }
            
            // Boost extra si hay coincidencia exacta de la frase entera
            if (chunk.text.toLowerCase().includes(queryLower)) {
                score += 0.2;
            }
            if (chunk.entities && chunk.entities.some(e => e.toLowerCase().includes(queryLower))) {
                score += 0.1;
            }
            
            return {
                ...chunk,
                score
            };
        });
        
        // 3. Ordenar por mayor puntuación y aplicar umbral
        scoredChunks.sort((a, b) => b.score - a.score);
        // Umbral bajado a 0.15 porque las similitudes de vectores suelen rondar el 0.2
        const topResults = scoredChunks.filter(r => r.score >= 0.15).slice(0, 5);

        if (topResults.length === 0) {
            resultsEl.innerHTML = "<div style='color:#a83232; margin-top:20px;'>No se encontraron resultados relevantes que superen el umbral de similitud. Prueba a reformular la pregunta.</div>";
            statusEl.textContent = "Búsqueda completada (Sin resultados relevantes).";
            return;
        }
        
        // Mostrar resultados en la lista
        topResults.forEach(res => {
            const div = document.createElement('div');
            div.className = 'result';

            // Nombre de fichero como enlace clicable al visor de transcripciones
            const fileDiv = document.createElement('div');
            fileDiv.className = 'filename';
            const fileLink = document.createElement('a');
            const entParams = (res.entities || []).join('|');
            fileLink.href   = `viewer.html?file=${encodeURIComponent(res.filename)}&entities=${encodeURIComponent(entParams)}`;
            fileLink.target = '_blank';
            fileLink.rel    = 'noopener noreferrer';
            fileLink.textContent = `📄 ${res.filename}`;
            const scoreSpan = document.createElement('span');
            scoreSpan.style.marginLeft = '8px';
            scoreSpan.style.opacity    = '0.7';
            scoreSpan.textContent = `(Similitud: ${(res.score * 100).toFixed(1)}%)`;
            fileDiv.appendChild(fileLink);
            fileDiv.appendChild(scoreSpan);

            // Texto del chunk con entidades resaltadas
            const textDiv = document.createElement('div');
            textDiv.className = 'chunk-text';
            textDiv.innerHTML = highlightEntities(res.text, res.entities);

            const entDiv = document.createElement('div');
            entDiv.className = 'entities';
            if (res.entities && res.entities.length) {
                entDiv.innerHTML = '<strong>Entidades detectadas:</strong> ' +
                    res.entities.map(e => `<span class="entity-tag">${escapeHtml(e)}</span>`).join(' ');
            }

            div.appendChild(fileDiv);
            div.appendChild(textDiv);
            div.appendChild(entDiv);
            resultsEl.appendChild(div);
        });
        
        // Dibujar Grafo de entidades D3.js
        drawGraph(topResults);
        
        // Generar respuesta con WebLLM
        statusEl.textContent = "Generando respuesta con Qwen2.5...";
        llmEl.style.display = "block";
        llmEl.textContent = "Pensando...";
        
        const context = topResults.map(r => r.text).join("\n\n");
        const prompt = `Contexto:\n${context}\n\nPregunta: ${query}\nRespuesta:`;
        
        const messages = [
            { role: "system", content: "Eres un asistente geopolítico. Responde a la pregunta del usuario utilizando de forma clara y objetiva SÓLO la información proporcionada en el Contexto. Si la información no está en el contexto, indica que no lo sabes." },
            { role: "user", content: prompt }
        ];
        
        // Respuesta en streaming para efecto "escribiendo"
        const asyncChunkGenerator = await engine.chat.completions.create({
            messages,
            stream: true,
            temperature: 0.2, // Baja temperatura para RAG factico
        });
        
        llmEl.textContent = "";
        for await (const chunk of asyncChunkGenerator) {
            llmEl.textContent += chunk.choices[0]?.delta?.content || "";
        }
        
        statusEl.textContent = `Búsqueda y generación completadas.`;
        
    } catch (e) {
        statusEl.textContent = `Error en la búsqueda/generación: ${e.message}`;
    } finally {
        document.getElementById('searchBtn').disabled = false;
    }
}

document.getElementById('searchBtn').addEventListener('click', search);
document.getElementById('searchInput').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') search();
});

// Arrancar inicialización al cargar el script
init();
