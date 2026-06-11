import { pipeline } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.1';
import * as webllm from "https://esm.run/@mlc-ai/web-llm";

let extractor;
let database = [];
let engine;

// Función para calcular similitud del coseno entre dos vectores
function cosineSimilarity(vecA, vecB) {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    for (let i = 0; i < vecA.length; i++) {
        dotProduct += vecA[i] * vecB[i];
        normA += vecA[i] * vecA[i];
        normB += vecB[i] * vecB[i];
    }
    if (normA === 0 || normB === 0) return 0;
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

async function init() {
    const statusEl = document.getElementById('status');
    
    try {
        // 1. Cargar el mismo modelo usado en Python
        statusEl.textContent = "Cargando modelo de embeddings (WASM)...";
        extractor = await pipeline('feature-extraction', 'Xenova/paraphrase-multilingual-MiniLM-L12-v2');
        
        // 2. Cargar la base de datos de vectores
        statusEl.textContent = "Cargando base de datos vectorial...";
        const response = await fetch('database.json');
        if (!response.ok) throw new Error("No se encontró database.json. ¿Has ejecutado indexer.py en el servidor?");
        database = await response.json();
        
        // 3. Cargar WebLLM (Qwen)
        statusEl.textContent = "Cargando WebLLM (Qwen2.5-0.5B)... Esto puede tardar varios minutos y descargar ~350MB la primera vez.";
        
        const initProgressCallback = (initProgress) => {
            statusEl.textContent = `Descargando LLM: ${initProgress.text}`;
        }
        
        engine = await webllm.CreateMLCEngine(
            "Qwen2.5-0.5B-Instruct-q4f16_1-MLC", 
            { initProgressCallback }
        );
        
        // 4. Listo para buscar
        statusEl.textContent = `¡Listo! Base de datos cargada con ${database.length} fragmentos de transcripciones.`;
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

// Dibujar grafo de fuerzas D3.js con las entidades de los resultados
function drawGraph(topResults) {
    const svgEl = document.getElementById('graph');
    svgEl.style.display = 'block';
    svgEl.innerHTML = ''; // Limpiar SVG anterior
    
    const width = svgEl.clientWidth;
    const height = svgEl.clientHeight || 300;
    
    const nodesMap = new Map();
    const linksMap = new Map();
    
    // Crear nodos y enlaces a partir de las co-ocurrencias en los chunks
    topResults.forEach(res => {
        const ents = res.entities || [];
        ents.forEach(e => {
            if (!nodesMap.has(e)) nodesMap.set(e, { id: e, group: 1 });
        });
        
        for (let i = 0; i < ents.length; i++) {
            for (let j = i + 1; j < ents.length; j++) {
                const source = ents[i];
                const target = ents[j];
                const linkId = [source, target].sort().join('-');
                if (!linksMap.has(linkId)) {
                    linksMap.set(linkId, { source, target, value: 1 });
                } else {
                    linksMap.get(linkId).value++;
                }
            }
        }
    });
    
    const nodes = Array.from(nodesMap.values());
    const links = Array.from(linksMap.values());
    
    if (nodes.length === 0) {
        svgEl.style.display = 'none';
        return;
    }

    const svg = d3.select("#graph")
        .attr("viewBox", [0, 0, width, height]);
        
    const simulation = d3.forceSimulation(nodes)
        .force("link", d3.forceLink(links).id(d => d.id).distance(100))
        .force("charge", d3.forceManyBody().strength(-200))
        .force("center", d3.forceCenter(width / 2, height / 2));

    const link = svg.append("g")
        .selectAll("line")
        .data(links)
        .join("line")
        .attr("class", "link")
        .attr("stroke-width", d => Math.sqrt(d.value));

    const node = svg.append("g")
        .selectAll("g")
        .data(nodes)
        .join("g");

    node.append("circle")
        .attr("r", 10)
        .attr("fill", "#0066cc")
        .style("cursor", "pointer")
        .on("click", (event, d) => {
            document.getElementById('searchInput').value = d.id;
            search(); // Lanzar búsqueda al hacer click
        })
        .call(d3.drag()
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended));

    node.append("text")
        .text(d => d.id)
        .attr("x", 12)
        .attr("y", 3);

    simulation.on("tick", () => {
        link
            .attr("x1", d => d.source.x)
            .attr("y1", d => d.source.y)
            .attr("x2", d => d.target.x)
            .attr("y2", d => d.target.y);

        node
            .attr("transform", d => `translate(${d.x},${d.y})`);
    });

    function dragstarted(event, d) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }
    function dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }
    function dragended(event, d) {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }
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
        
        // 2. Calcular la similitud contra todos los chunks de la base de datos
        const scoredChunks = database.map(chunk => {
            return {
                ...chunk,
                score: cosineSimilarity(queryEmbedding, chunk.embedding)
            };
        });
        
        // 3. Ordenar por mayor puntuación y mostrar los top 5
        scoredChunks.sort((a, b) => b.score - a.score);
        const topResults = scoredChunks.slice(0, 5);
        
        // Mostrar resultados en la lista
        topResults.forEach(res => {
            const div = document.createElement('div');
            div.className = 'result';
            
            const fileDiv = document.createElement('div');
            fileDiv.className = 'filename';
            fileDiv.textContent = `📄 ${res.filename} (Similitud: ${(res.score * 100).toFixed(1)}%)`;
            
            const textDiv = document.createElement('div');
            textDiv.textContent = `"...${res.text}..."`;
            
            const entDiv = document.createElement('div');
            entDiv.className = 'entities';
            entDiv.textContent = res.entities && res.entities.length ? `Entidades: ${res.entities.join(", ")}` : "";
            
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
