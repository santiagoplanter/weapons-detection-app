import streamlit as st
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# 🔹 Configuración de la página en Streamlit
st.set_page_config(page_title="Weapons Detection", page_icon="🔫", layout="wide")

# 🔹 Cargar modelo YOLO
DEVICE = "cpu"
MODEL_PATH = "big_model_yolov11_knife.pt"
model = YOLO(MODEL_PATH).to(DEVICE)

# 🔹 Función para dibujar detecciones
def draw_detections(image, results):
    """Dibuja cajas y etiquetas en la imagen."""
    for result in results:
        for box in result.boxes:
            confidence = box.conf[0].item()
            if confidence >= 0.5:
                x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                label = f"{model.names[class_id]}: {confidence:.2f}"
                
                # Dibujar bounding box y etiqueta
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
                cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
    
    return image

# 🔹 Header del sitio
st.image("images/title.png", use_container_width=True)
st.image("images/line.png")

# 🔹 Subida de imágenes
st.markdown("<h1 style='text-align: center;'>UPLOAD OR TAKE A PICTURE</h1>", unsafe_allow_html=True)
uploaded_image = st.file_uploader("", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Convertir imagen a RGB (para PNG y JPG)
    image = Image.open(uploaded_image).convert("RGB")
    image = np.array(image)

    # Convertir imagen a tensor para YOLO
    image_tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    image_tensor = image_tensor.to(DEVICE)

    # Ejecutar modelo
    results = model(image_tensor)

    # Dibujar detecciones
    image_with_boxes = draw_detections(image.copy(), results)

    # Mostrar imagen con detecciones
    st.image(image_with_boxes, caption="Detected Image", use_column_width=True)

st.image("images/line.png")

# 🔹 Webcam con detección en vivo
st.markdown("<h1 style='text-align: center;'>TRY WITH LIVE WEBCAM</h1>", unsafe_allow_html=True)

class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Convertir imagen a tensor
        image_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        image_tensor = image_tensor.to(DEVICE)

        # Ejecutar detección
        results = model(image_tensor)
        img_with_boxes = draw_detections(img.copy(), results)

        return av.VideoFrame.from_ndarray(img_with_boxes, format="bgr24")

# 🔹 Activar webcam con botón
if "camera_active" not in st.session_state:
    st.session_state["camera_active"] = False

if st.button("Start Live Camera"):
    st.session_state["camera_active"] = not st.session_state["camera_active"]

if st.session_state["camera_active"]:
    webrtc_streamer(
        key="example",
        video_processor_factory=VideoProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": {"width": {"ideal": 640}, "height": {"ideal": 480}, "frameRate": {"ideal": 15}}, "audio": False},
    )

st.image('images/line.png')
