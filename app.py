import io, os, tempfile, shutil
from pathlib import Path
from PIL import Image
import streamlit as st
from ultralytics import YOLO

st.set_page_config(page_title="BJ Card Counter (YOLO) Demo", layout="centered")
st.title("🃏 BJ Card Counter — YOLO Demo")

with st.sidebar:
    st.subheader("Model")
    weights = st.text_input(
        "Weights path or model name:",
        value="yolov8n.pt",  # change to "best.pt" if you want yours by default
        help="Use a local path like best.pt (upload it) or a built-in model like yolov8n.pt"
    )
    conf = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.01)

@st.cache_resource(show_spinner=False)
def load_model(w):
    return YOLO(w)

# Try loading model (shows a nice error if missing)
try:
    model = load_model(weights)
except Exception as e:
    st.error(f"Failed to load model '{weights}': {e}")
    st.stop()

tab_img, tab_vid = st.tabs(["🖼️ Image", "🎞️ Video"])

with tab_img:
    file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if file:
        img = Image.open(file).convert("RGB")
        st.image(img, caption="Input", use_container_width=True)
        with st.spinner("Running inference…"):
            results = model.predict(img, conf=conf, verbose=False)
            plotted_bgr = results[0].plot()
        plotted_rgb = plotted_bgr[:, :, ::-1]
        st.image(plotted_rgb, caption="Detections", use_container_width=True)

        # Download
        buf = io.BytesIO()
        Image.fromarray(plotted_rgb).save(buf, format="JPEG", quality=95)
        st.download_button("Download image result", data=buf.getvalue(),
                           file_name="prediction.jpg", mime="image/jpeg")
    else:
        st.info("Upload a JPG/PNG to run detection.")

with tab_vid:
    vfile = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
    if vfile:
        # Save upload to a real temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(vfile.name).suffix) as tmp_in:
            tmp_in.write(vfile.read())
            in_path = Path(tmp_in.name)

        st.video(str(in_path))  # preview original
        with st.spinner("Running video inference… (this can take a bit)"):
            # Run Ultralytics inference with save=True so it writes an annotated video
            results = model.predict(
                source=str(in_path),
                conf=conf,
                save=True,
                verbose=False
            )
            # Ultralytics saves to runs/detect/predict*, find the output path
            # results[0].save_dir is a pathlib Path to the run directory
            save_dir = Path(results[0].save_dir)
            # Find first video file in save_dir (should match input name)
            outs = list(save_dir.glob("*.*"))
            out_video = None
            for p in outs:
                if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
                    out_video = p
                    break

        if out_video and out_video.exists():
            st.success("Done!")
            st.video(str(out_video))
            with open(out_video, "rb") as f:
                st.download_button(
                    "Download video result",
                    data=f.read(),
                    file_name=out_video.name,
                    mime="video/mp4"
                )
        else:
            st.error("Could not locate the annotated video. Make sure ffmpeg is installed.")
    else:
        st.info("Upload an MP4/MOV/AVI/MKV for detection.")
