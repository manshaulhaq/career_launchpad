import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import imageio
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Helmet Detection System", layout="wide")
st.title("Real-Time Helmet Compliance System")

# --- PATH CONFIG ---
MODEL_PATH = 'runs/detect/train/weights/best.pt'

# --- SIDEBAR SETTINGS ---
st.sidebar.header("Control Panel")
# Setting optimal 0.48 threshold from Phase 5 report
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.48)
save_output = st.sidebar.checkbox("Record Session as GIF")

# Load Model
@st.cache_resource
def load_model(path):
    if os.path.exists(path):
        return YOLO(path)
    return None

model = load_model(MODEL_PATH)

# Initialize Session State
if 'run_system' not in st.session_state:
    st.session_state.run_system = False

def toggle_system():
    st.session_state.run_system = not st.session_state.run_system

# --- MAIN UI ---
if model:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Live Feed")
        frame_placeholder = st.empty()
        # Toggle button
        if not st.session_state.run_system:
            st.button("Start System", on_click=toggle_system)
        else:
            st.button("Stop System", on_click=toggle_system)

    with col2:
        st.subheader("System Status")
        status_text = st.empty()
        if st.session_state.run_system:
            st.success("System Live")
            if save_output:
                st.warning("Recording Active...")
        else:
            st.info("System Standby")

    # --- PROCESSING LOOP ---
    if st.session_state.run_system:
        # Try to open camera with a small delay or retry logic
        cap = cv2.VideoCapture(0)
        
        # Performance fix: set lower resolution for smoother live feed
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        frames_for_gif = []

        while st.session_state.run_system:
            ret, frame = cap.read()
            if not ret:
                # If camera fails, stop the system state and break
                st.session_state.run_system = False
                st.error("Webcam connection lost or not detected.")
                break

            # Inference
            results = model.predict(frame, conf=conf_threshold, verbose=False)
            annotated_frame = results[0].plot()
            
            # Display conversion
            display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(display_frame, channels="RGB", use_container_width=True)
            
            # Update metrics
            count = len(results[0].boxes)
            status_text.metric(label="Active Detections", value=count)

            # Record frames
            if save_output:
                # Downsample for GIF storage efficiency
                small_frame = cv2.resize(display_frame, (480, 360))
                frames_for_gif.append(small_frame)

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Save GIF only if system was stopped and frames exist
        if save_output and len(frames_for_gif) > 0:
            with st.spinner("Writing GIF file..."):
                imageio.mimsave('detection_output.gif', frames_for_gif, fps=10)
                st.success("Session saved: detection_output.gif")
else:
    st.error("Model file missing.")
