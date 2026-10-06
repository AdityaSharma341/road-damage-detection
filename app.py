"""
Road Damage Detection - Streamlit Web App
Built around a YOLOv8 model trained in train.ipynb (ultralytics).

Run with:
    streamlit run app.py
"""

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Road Damage Detection",
    page_icon="🛣️",
    layout="wide",
)

# --------------------------------------------------------------------------
# Fixed model & inference settings (no sidebar controls)
# --------------------------------------------------------------------------
DEFAULT_WEIGHTS = "runs/detect/runs/road_damage_detection_v2-2/weights/best.pt"

weights_path = DEFAULT_WEIGHTS
uploaded_weights = None
conf_threshold = 0.25
iou_threshold = 0.45
img_size = 960


# --------------------------------------------------------------------------
# Load model (cached so it isn't reloaded on every interaction)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading YOLO model...")
def load_model(path: str):
    return YOLO(path)


def get_model():
    # Priority: uploaded weights file > text path
    if uploaded_weights is not None:
        tmp_dir = tempfile.mkdtemp()
        tmp_path = os.path.join(tmp_dir, uploaded_weights.name)
        with open(tmp_path, "wb") as f:
            f.write(uploaded_weights.getbuffer())
        return load_model(tmp_path)

    if not weights_path or not Path(weights_path).exists():
        return None

    return load_model(weights_path)


model = get_model()

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("🛣️ Road Damage Detection")
st.write(
    "Upload road images (or a short video) and this app will run your "
    "trained YOLOv8 model to detect and localize road damage."
)

if model is None:
    st.warning(
        f"⚠️ Could not find a valid model at `{DEFAULT_WEIGHTS}`. "
        "Update the `DEFAULT_WEIGHTS` path at the top of app.py to point "
        "to your trained `best.pt` file."
    )
    st.stop()
else:
    st.success(f"Model loaded ✅  |  Classes: {list(model.names.values())}")

st.markdown("---")

# --------------------------------------------------------------------------
# Tabs: Image inference / Video inference
# --------------------------------------------------------------------------
tab_image, tab_video, tab_about = st.tabs(["📷 Image", "🎞️ Video", "ℹ️ About"])

# ---------------- Image tab ----------------
with tab_image:
    uploaded_images = st.file_uploader(
        "Upload one or more images",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True,
        key="image_uploader",
    )

    if uploaded_images:
        run_btn = st.button("🚀 Run Detection", type="primary", key="run_image")

        if run_btn:
            for uploaded_file in uploaded_images:
                image = Image.open(uploaded_file).convert("RGB")
                img_array = np.array(image)

                with st.spinner(f"Detecting damage in {uploaded_file.name}..."):
                    results = model.predict(
                        source=img_array,
                        conf=conf_threshold,
                        iou=iou_threshold,
                        imgsz=img_size,
                        verbose=False,
                    )

                result = results[0]
                annotated = result.plot()  # BGR numpy array with boxes drawn
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

                st.subheader(uploaded_file.name)
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="Original", use_container_width=True)
                with col2:
                    st.image(
                        annotated_rgb, caption="Detections", use_container_width=True
                    )

                # Detection table
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    data = []
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        data.append(
                            {
                                "class": model.names[cls_id],
                                "confidence": round(float(box.conf[0]), 3),
                                "x1": round(float(box.xyxy[0][0]), 1),
                                "y1": round(float(box.xyxy[0][1]), 1),
                                "x2": round(float(box.xyxy[0][2]), 1),
                                "y2": round(float(box.xyxy[0][3]), 1),
                            }
                        )
                    df = pd.DataFrame(data)
                    st.dataframe(df, use_container_width=True)

                    # Download annotated image
                    result_pil = Image.fromarray(annotated_rgb)
                    buf_path = os.path.join(
                        tempfile.mkdtemp(), f"detected_{uploaded_file.name}"
                    )
                    result_pil.save(buf_path)
                    with open(buf_path, "rb") as f:
                        st.download_button(
                            "⬇️ Download annotated image",
                            data=f,
                            file_name=f"detected_{uploaded_file.name}",
                            mime="image/png",
                            key=f"dl_{uploaded_file.name}",
                        )
                else:
                    st.info("No damage detected above the confidence threshold.")

                st.markdown("---")
    else:
        st.info("Upload image(s) above to get started.")

# ---------------- Video tab ----------------
with tab_video:
    uploaded_video = st.file_uploader(
        "Upload a video", type=["mp4", "avi", "mov", "mkv"], key="video_uploader"
    )

    if uploaded_video is not None:
        run_video_btn = st.button("🚀 Run Detection on Video", type="primary")

        if run_video_btn:
            tmp_dir = tempfile.mkdtemp()
            in_path = os.path.join(tmp_dir, uploaded_video.name)
            with open(in_path, "wb") as f:
                f.write(uploaded_video.getbuffer())

            cap = cv2.VideoCapture(in_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            out_path = os.path.join(tmp_dir, "annotated_" + uploaded_video.name)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

            progress = st.progress(0, text="Processing video...")
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                results = model.predict(
                    source=frame,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    imgsz=img_size,
                    verbose=False,
                )
                annotated_frame = results[0].plot()
                writer.write(annotated_frame)

                frame_idx += 1
                if total_frames > 0:
                    progress.progress(
                        min(frame_idx / total_frames, 1.0),
                        text=f"Processing frame {frame_idx}/{total_frames}",
                    )

            cap.release()
            writer.release()
            progress.empty()

            st.success("Done! Preview and download below.")
            st.video(out_path)
            with open(out_path, "rb") as f:
                st.download_button(
                    "⬇️ Download annotated video",
                    data=f,
                    file_name="annotated_" + uploaded_video.name,
                    mime="video/mp4",
                )
    else:
        st.info("Upload a video above to get started.")

# ---------------- About tab ----------------
with tab_about:
    st.markdown(
        """
        ### About this app
        This interface wraps a **YOLOv8** (`ultralytics`) object detection
        model trained for **road damage detection** (potholes, cracks, etc.,
        depending on your `Data.yaml` class list).

        **How to use:**
        1. Make sure `DEFAULT_WEIGHTS` at the top of `app.py` points to your
           trained `best.pt` file.
        2. Upload an image or video on the corresponding tab and click
           **Run Detection**.
        3. Review results and download the annotated output.

        **Model classes detected:**
        """
    )
    if model is not None:
        st.write(list(model.names.values()))