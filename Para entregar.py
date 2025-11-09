### 1. Importaciones (líneas iniciales)
```python
# app.py - GUARDA IMÁGENES EN CLOUDINARY (CARPETA CLASIFICADOR_FRUTAS) + METADATA EN SUPABASE DB + ESTADÍSTICAS + ELIMINAR
from flask import Flask, render_template, request, jsonify, send_from_directory
from ultralytics import YOLO
import cloudinary
import cloudinary.uploader
import cv2
import numpy as np
import os
import base64
from datetime import datetime
from supabase import create_client, Client
```
- **Comentario inicial**: Es solo una descripción del archivo. Indica que guarda imágenes en Cloudinary en una carpeta específica, metadata en Supabase, calcula stats, y permite eliminar.
- **Flask imports**: Importo lo básico de Flask para crear la app, renderizar templates HTML, manejar requests (como POST para subir archivos), devolver JSON, y servir archivos estáticos.
- **ultralytics.YOLO**: Para cargar y usar el modelo de detección de objetos (YOLOv8 en este caso, entrenado para frutas).
- **cloudinary y cloudinary.uploader**: Para subir imágenes a Cloudinary (un servicio de almacenamiento en la nube para medios).
- **cv2 (OpenCV)**: Para procesar imágenes, como decodificar, resize, encode.
- **numpy as np**: Para manejar arrays de imágenes (OpenCV usa numpy internamente).
- **os**: Para chequear si archivos existen (como el modelo).
- **base64**: Para decodificar imágenes que vienen en base64 (de la cámara).
- **datetime**: Para generar timestamps.
- **supabase**: Para conectar y operar con la base de datos de Supabase (un backend como servicio, similar a Firebase).

Esto configura todo lo que necesito. Sin estas imports, nada funcionaría.

### 2. Creación de la app Flask
```python
app = Flask(__name__, template_folder='templates', static_folder='static')
```
- Creo la instancia de Flask llamada `app`. Específico las carpetas para templates (HTML) y static (CSS, JS, sonidos, etc.). Esto es estándar para que Flask sepa dónde buscar archivos.

### 3. Configuración de Cloudinary
```python
# === CLOUDINARY ===
cloudinary.config(
    cloud_name=
    api_key=
    api_secret=
)
```
- Configuro Cloudinary con mi cloud_name, api_key y api_secret. (Nota: En esta explicación, no las repito por seguridad – en tu código real, reemplázalas con variables de entorno como `os.getenv('CLOUDINARY_API_KEY')` para no exponerlas).
- Esto permite subir archivos a mi cuenta de Cloudinary. Sin esto, las subidas fallarían.

### 4. Configuración de Supabase
```python
# === SUPABASE ===
SUPABASE_URL =
SUPABASE_KEY =
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
```
- Defino la URL de mi proyecto Supabase y la key (de nuevo, no la repito aquí – úsala con cuidado, es como una contraseña maestra).
- Creo un cliente Supabase para interactuar con la DB (tabla "biblioteca"). Esto me permite insertar, seleccionar, eliminar registros.

### 5. Carga del modelo YOLO
```python
# === MODELO ===
MODEL_PATH = "modelo/best_yolov8s_fruits_v1.pt"
if not os.path.exists(MODEL_PATH):
    print("ERROR: Modelo no encontrado")
    exit()
model = YOLO(MODEL_PATH)
print("Modelo cargado")
```
- Defino la ruta al archivo del modelo (.pt es el formato de YOLO entrenado).
- Chequeo si existe con `os.path.exists`; si no, imprimo error y salgo del programa (para evitar crashes más adelante).
- Cargo el modelo con `YOLO()`. Imprimo "Modelo cargado" para confirmar en consola.
- Este modelo está entrenado para detectar frutas (asumiendo clases como manzanas, bananas, etc., en `model.names`).

### 6. Rutas de Flask (sección principal)
```python
# === RUTAS ===
```
- Aquí empiezan las rutas (endpoints) de la app. Cada `@app.route` define una URL y qué hacer cuando se accede.

#### Ruta para servir el logo
```python
@app.route('/logo.png')
def serve_logo():
    return send_from_directory('.', 'logo.png')
```
- Sirve el archivo 'logo.png' desde el directorio actual. Útil si el HTML lo referencia directamente.

#### Ruta para servir sonidos
```python
@app.route('/static/sounds/<path:filename>')
def serve_sounds(filename):
    return send_from_directory('static/sounds', filename)
```
- Sirve archivos de sonido desde 'static/sounds'. El `<path:filename>` captura el nombre del archivo en la URL.

#### Ruta principal (index)
```python
@app.route("/")
def index():
    return render_template("index.html")
```
- Cuando accedes a '/', renderiza 'index.html' (el frontend, probablemente con JS para subir imágenes o capturar cámara).

#### Ruta para subir imagen (upload)
```python
@app.route("/upload", methods=["POST"])
def upload():
    try:
        file = request.files['imagen']
        if not file or file.filename == '':
            return jsonify({"error": "No imagen"}), 400
        img_bytes = file.read()
        npimg = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Imagen inválida"}), 400
        # Procesar con YOLO
        results = model(img, conf=0.4, verbose=False)
        annotated = results[0].plot()
        # Detecciones
        detecciones = []
        has_detection = False
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            has_detection = True
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = round(float(box.conf[0].item()), 2)
                clase = model.names[cls_id]
                detecciones.append({"clase": clase, "conf": conf})
        # Calcular promedio de confianza
        confidence_average = sum(d['conf'] for d in detecciones) / len(detecciones) if detecciones else 0
        detection_count = len(detecciones)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Subir imagen procesada a Cloudinary
        _, annotated_buffer = cv2.imencode('.jpg', annotated)
        result = cloudinary.uploader.upload(
            annotated_buffer.tobytes(),
            folder="CLASIFICADOR_FRUTAS",
            public_id=f"proc_{timestamp}",
            format="jpg"
        )
        url = result['secure_url']
        # Generar y subir thumbnail a Cloudinary
        thumb_img = cv2.resize(img, (200, 200))
        _, thumb_buffer = cv2.imencode('.jpg', thumb_img)
        thumb_result = cloudinary.uploader.upload(
            thumb_buffer.tobytes(),
            folder="CLASIFICADOR_FRUTAS",
            public_id=f"thumb_{timestamp}",
            format="jpg"
        )
        thumbnail_url = thumb_result['secure_url']
        # Guardar metadata en Supabase
        entry = {
            "timestamp": datetime.now().strftime("%d/%m/%Y %H:%M"),
            "detecciones": detecciones,
            "url": url,
            "thumbnail_url": thumbnail_url,
            "has_detection": has_detection,
            "original_filename": file.filename,
            "source": "upload",
            "confidence_average": confidence_average,
            "detection_count": detection_count
        }
        response = supabase.table("biblioteca").insert(entry).execute()
        entry["id"] = response.data[0]["id"]  # Obtener el ID asignado por Supabase
        return jsonify(entry)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
```
- Esta es la ruta principal para subir una imagen desde un archivo.
- **Try-except**: Envuelvo todo en try para capturar errores y devolver 500 con el mensaje.
- Obtengo el archivo de `request.files['imagen']`. Si no hay, error 400.
- Leo los bytes, convierto a numpy array, decodifico con cv2 a imagen color.
- Si falla, error 400.
- Proceso con YOLO: `model(img, conf=0.4, verbose=False)` – conf=0.4 significa detecciones con al menos 40% confianza, sin logs verbose.
- `annotated = results[0].plot()`: Dibuja boxes en la imagen.
- Extraigo detecciones: Para cada box, obtengo clase (de model.names), confianza redondeada, y las guardo en lista.
- `has_detection`: True si hay al menos una detección.
- Calculo promedio de confianza y conteo de detecciones.
- Genero timestamp único (año mes día_hora min seg).
- Encodeo la imagen anotada a JPG buffer y subo a Cloudinary: En carpeta "CLASIFICADOR_FRUTAS", con public_id como "proc_timestamp".
- Obtengo URL segura.
- Creo thumbnail: Resize a 200x200, encodeo, subo similarmente con "thumb_timestamp".
- Creo diccionario `entry` con metadata: timestamp legible, detecciones, URLs, etc. Fuente es "upload".
- Inserto en Supabase tabla "biblioteca", obtengo el ID autogenerado y lo agrego a entry.
- Devuelvo entry como JSON.

Esto maneja subidas de archivos.

#### Ruta para captura de cámara
```python
@app.route("/captura", methods=["POST"])
def captura():
    # Código similar a upload, pero con diferencias mínimas
    ...
```
- Muy similar a /upload, pero:
- Recibe JSON con 'imagen' en base64 (de la cámara web, probablemente via JS).
- Decodifica base64: `base64.b64decode(data['imagen'].split(',')[1])` – salta el header "data:image/jpeg;base64,".
- Resto igual: Procesa, sube, guarda metadata.
- Fuente es "camera", filename es "capture_timestamp.jpg".
- Diferencia clave: Input es base64 en vez de archivo multipart.

#### Ruta para obtener biblioteca
```python
@app.route("/biblioteca")
def get_biblioteca():
    try:
        res = supabase.table("biblioteca").select("*").order("created_at", desc=True).execute()
        biblioteca = res.data
        return jsonify(biblioteca)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
```
- Selecciona todos los registros de "biblioteca", ordenados por created_at descendente (más nuevos primero).
- Devuelve como JSON. Simple para listar en el frontend.

#### Ruta para eliminar registro
```python
@app.route("/delete/<int:db_id>", methods=["DELETE"])
def delete_record(db_id):
    try:
        # Obtener el registro de Supabase
        res = supabase.table("biblioteca").select("url, thumbnail_url").eq("id", db_id).execute()
        if not res.data:
            return jsonify({"error": "Registro no encontrado"}), 404
        record = res.data[0]
        url = record["url"]
        thumbnail_url = record["thumbnail_url"]
        # Extraer public_id de las URLs de Cloudinary
        url_parts = url.split('/')
        thumbnail_parts = thumbnail_url.split('/')
        proc_public_id = f"CLASIFICADOR_FRUTAS/{url_parts[-1].split('.')[0]}"  # CLASIFICADOR_FRUTAS/proc_20251029_083412
        thumb_public_id = f"CLASIFICADOR_FRUTAS/{thumbnail_parts[-1].split('.')[0]}"  # CLASIFICADOR_FRUTAS/thumb_20251029_083412
        # Eliminar imágenes de Cloudinary
        cloudinary.uploader.destroy(proc_public_id)
        cloudinary.uploader.destroy(thumb_public_id)
        # Eliminar registro de Supabase
        supabase.table("biblioteca").delete().eq("id", db_id).execute()
        return jsonify({"message": "Registro e imágenes eliminados correctamente"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
```
- Para DELETE /delete/<id>, como /delete/123.
- Obtiene URLs del registro en Supabase por ID. Si no existe, 404.
- Extrae public_id de las URLs: Split por '/', toma el último parte sin extensión, y prependea la carpeta.
- Usa `destroy` para borrar las imágenes en Cloudinary.
- Borra el registro en Supabase.
- Devuelve mensaje de éxito. Esto asegura que al eliminar, se limpien tanto DB como storage.

#### Ruta para estadísticas
```python
@app.route("/estadisticas")
def get_estadisticas():
    try:
        res = supabase.table("biblioteca").select("*").execute()
        biblioteca = res.data
        total = len(biblioteca)
        detectadas = sum(1 for e in biblioteca if e["has_detection"])
        no_detectadas = total - detectadas
        conteo_clases = {}
        confidence_by_class = {}
        for entry in biblioteca:
            for d in entry["detecciones"]:
                clase = d["clase"]
                conf = d["conf"]
                conteo_clases[clase] = conteo_clases.get(clase, 0) + 1
                confidence_by_class[clase] = confidence_by_class.get(clase, []) + [conf]
        # Calcular promedio de confianza por clase
        avg_confidence_by_class = {
            clase: sum(confs) / len(confs) if confs else 0
            for clase, confs in confidence_by_class.items()
        }
        return jsonify({
            "total": total,
            "detectadas": detectadas,
            "no_detectadas": no_detectadas,
            "clases": conteo_clases,
            "avg_confidence_by_class": avg_confidence_by_class
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
```
- Obtiene todos los registros.
- Calcula: Total entradas, cuántas tienen detecciones, cuántas no.
- Conteo por clase: Loop por todas detecciones, suma en dict.
- Confianzas por clase: Lista de confs por clase.
- Promedio por clase: Suma / len.
- Devuelve JSON con stats. Útil para dashboards.

### 7. Ejecución de la app
```python
if __name__ == "__main__":
    print("\nPanel Admin: http://localhost:5000\n")
    app.run(host='127.0.0.1', port=5000, debug=False)
```
