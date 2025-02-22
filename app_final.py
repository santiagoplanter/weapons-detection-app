import streamlit as st
import cv2
import numpy as np
import tempfile
import torch
from ultralytics import YOLO
from PIL import Image
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase


# PAGE SET UP
# ===========
## Initialize our streamlit app
st.set_page_config(
    page_title="Weapons Detection", 
    page_icon='🔫',
    layout='wide'
    )

# Load trained YOLOv11 model
DEVICE = "cpu"
# ==========================
MODEL_PATH = "big_model_yolov11_knife.pt" 
model = YOLO(MODEL_PATH)

# DRAWING BOXES
# =============
def draw_detections(image, results):
    """Draw bounding boxes and labels on the image."""
    for result in results:
        for box in result.boxes:
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            confidence = box.conf[0].item()

            # Draw bounding box
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Display class label and confidence
            label = f"{model.names[class_id]}: {confidence:.2f}"
            cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return image


# FORMATTING
# ==========

# Set custom CSS styles
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@800&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=EB+Garamond:wght@800&display=swap');
    
    .header {
        font-family: 'Open Sans', sans-serif;
        font-size: 48px;
        font-weight: 600 !important;
        color: #333;
        margin-top: 50px;
    }

    .body {
        font-family: 'Montserrat', sans-serif;
        font-size: 20px;
        font-weight: 400 !important;
        color: #555;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# BUTTON CONFIGURATION
# ====================
st.markdown("""
    <style>
    .stButton > button {
        height: auto;
        padding-top: 20px;
        padding-bottom: 20px;
        font-weight: bold !important;
        color: white; /* white text in normal state */
        border: 5px solid black; /* black border */
        border-radius: 40px;
        width: 100%;
        cursor: pointer;
        background-color: black; /* black background in normal state */
    }
    .stButton > button:hover {
        background-color: white; /* white background on hover */
        color: black; /* pink text on hover */
        border: 5px solid black; /* Ensures border stays pink */
    }
    </style>
    """, unsafe_allow_html=True)


# WEBSITE HEADER
# ==============
st.image("images/title.png", use_container_width=True)

st.image("images/line.png")

# NAVIGATION BUTTONS
# ==================

col1, col2 = st.columns(2)

with col1:
    if st.button("HOME"):
        st.session_state.page = "HOME"    
with col2:
    if st.button("DEMO"):
        st.session_state.page = "DEMO" 

menu = ["HOME", "DEMO"]
# Initialize session state if not set

if 'page' not in st.session_state:
    st.session_state.page = "HOME"

# Show content based on the selected page
page = st.session_state.page

# HOME PAGE
# =========

if page == "HOME":
    st.write('')
    st.image('images/weaponsdetectionsolution.png')
    st.write('')
    st.write('')
    st.write('')

    # Display header and body text with inline styles for centering
    st.markdown("""
        <h1 style="font-family: \'Open Sans\', sans-serif; font-size: 48px; font-weight: 800; text-align: center">
            ENHANCING SECURITY WITH AI</h1>
            """, unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,6,1])
    with col1:
        st.empty()
    with col2:
        st.markdown("""
                <p style="font-family: \'Montserrat\', sans-serif; font-size: 20px; font-weight: 400; text-align: center">
                AI-powered weapons technology designed to enhance security and safety across various industries. Utilizing \
                YOLO V11 object detection models, our platform offers detection of pistols and knives in real-time. \
                Essential for security operations, public safety, and risk management.</p>
                """, unsafe_allow_html=True)
    with col3:
        st.empty()

    st.write('')
    st.write('')

    st.image('images/traindataset.png')
    

    st.write('')
    st.write('')
    st.write('')

# DEMO PAGE
# =========
if page == "DEMO":

    st.write('')
    st.write('')
    st.write('')

    st.image('images/mockup.png')

    st.write('')
    st.write('')
    st.write('---')

    # 📸 Image Upload
    st.markdown("""
        <h1 style="font-family: \'Open Sans\', sans-serif; font-size: 48px; font-weight: 800; text-align: center">
            UPLOAD OR TAKE A PICTURE</h1>
            """, unsafe_allow_html=True)    

    uploaded_image = st.file_uploader("", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        # Convertir la imagen a formato NumPy
        image = Image.open(uploaded_image).convert("RGB")  
        image = np.array(image)
    
        # Asegurar que la imagen es RGB
        if len(image.shape) == 2:  # Imagen en escala de grises
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
        # Convertir la imagen a un tensor de Torch
        image_tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        print("Imagen procesada correctamente:", image_tensor.shape)  # Para debug
    
        # Ejecutar modelo
        results = model(image_tensor)
    
        # Dibujar detecciones
        image_with_boxes = image.copy()
        for result in results:
            for box in result.boxes:
                confidence = box.conf[0].item()  # Obtener confianza
    
                if confidence >= 0.5:
                    x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
                    class_id = int(box.cls[0])
                    label = f"{model.names[class_id]}: {confidence:.2f}"
    
                    # Dibujar el cuadro delimitador y la etiqueta
                    cv2.rectangle(image_with_boxes, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
                    cv2.putText(image_with_boxes, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
        # Mostrar imagen con detecciones
        st.image(image_with_boxes, caption="Detected Image", use_column_width=True)
    
    st.image("images/line.png")
    
        # Mostrar la imagen con detecciones
        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            st.empty()
        with col2:
            st.image(image_with_boxes, caption="Detected Image", use_container_width=True)
        with col3:
            st.empty()


        # Draw detections only if confidence is greater than or equal to 0.5
        image_with_boxes = image.copy()
        for result in results:
            for box in result.boxes:
                confidence = box.conf[0].item()  # Get confidence score

                # Only draw the bounding box if the confidence is >= 0.5
                if confidence >= 0.5:
                    x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
                    class_id = int(box.cls[0])
                    label = f"{model.names[class_id]}: {confidence:.2f}"
                    
                    # Draw bounding box and label
                    cv2.rectangle(image_with_boxes, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
                    cv2.putText(image_with_boxes, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)

        col1, col2, col3 = st.columns([1,3,1])
        with col1:
            st.empty()
        with col2:
            # Display output image below the columns
            st.image(image_with_boxes, caption="Detected Image", use_container_width=True)
        with col3:
            st.empty()

    st.write('')
    st.write('')
    st.write('---')

    # 📹 Live Webcam Detection with Stop Button
    st.markdown("""
        <h1 style="font-family: 'Open Sans', sans-serif; font-size: 48px; font-weight: 800; text-align: center">
            TRY WITH LIVE WEBCAM</h1>
            """, unsafe_allow_html=True) 
    
    # 🔹 Define la función antes de llamarla


class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0  # Contador de frames

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")  # Convertir frame a NumPy array
        self.frame_count += 1  # Incrementar contador

        # Procesar YOLO solo cada 5 frames para mejorar rendimiento
        if self.frame_count % 5 == 0:
            results = model(img)

            for result in results:
                for box in result.boxes:
                    confidence = box.conf[0].item()
                    if confidence >= 0.5:
                        x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
                        class_id = int(box.cls[0])
                        label = f"{model.names[class_id]}: {confidence:.2f}"

                        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
                        cv2.putText(img, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# 🔹 Agregar un mensaje antes de iniciar la cámara
# Inicializar la variable en session_state si no existe
if "camera_active" not in st.session_state:
    st.session_state["camera_active"] = False  # La cámara empieza apagada

# Botón para activar/desactivar la cámara
if st.button("Start Live Camera"):
    st.session_state["camera_active"] = not st.session_state["camera_active"]  # Alternar estado

# Mostrar la cámara solo si está activa
if st.session_state["camera_active"]:
    st.write("Iniciando WebRTC...")
    webrtc_streamer(
        key="example",
        video_processor_factory=VideoProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640},  # Reducir resolución a 640x480 para mejorar velocidad
                "height": {"ideal": 480},
                "frameRate": {"ideal": 15}  # Reducir FPS para evitar lag
            },
            "audio": False,
        },
    )



st.image('images/line.png')
