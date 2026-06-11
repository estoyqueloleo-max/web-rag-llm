Me dejas en el README de web-rag estas configuraiconeS? lincamos en la web a la seccion del readme?

y si hacemos que los nodos sean lincables y relancen la busqueda? para ver como avanza?

ponemos en el readme todo el proceso?

y lo lanzamos con todo la carpeta .... que deberiamos mover dentro de la carpeta web... a ver como salen de tamaño los ficheros?

Y vamos preparando un github pages para mover la carpeta a un repo nuevo?

---

## 🔧 Tareas Pendientes a Futuro / Troubleshooting
- **Arreglar IPEX (`intel_extension_for_pytorch`) en Linux**:
  Al intentar usar la GPU Intel para inferencia o entrenamiento pesado en PyTorch, la librería (`libintel-ext-pt-cpu.so`) falla con el error de seguridad: 
  `cannot enable executable stack as shared object requires: Invalid argument`.
  **Solución**: Instalar la utilidad `execstack` y limpiar la bandera de seguridad de la librería ejecutando `sudo execstack -c /ruta/a/libintel-ext-pt-cpu.so`, o reinstalar una versión que sea compatible con las políticas del kernel.