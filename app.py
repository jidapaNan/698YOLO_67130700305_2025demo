import streamlit as st
from ultralytics import YOLO
from PIL import Image, UnidentifiedImageError
import numpy as np
import io

st.title("YOLO Image Detection App :)")

@st.cache_resource
def load_model():
    return YOLO("best.pt")
model = load_model()

uploaded = st.file_uploader(
    "Upload an image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"]
)

def is_jpeg_or_png(data: bytes) -> bool:
    # JPEG starts with FF D8 FF, PNG starts with 89 50 4E 47 0D 0A 1A 0A
    if data.startswith(b"\xff\xd8\xff"):
        return True
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return True
    return False

if uploaded is not None:
    try:
        # อ่านเป็น bytes เสมอ แล้ว reset pointer ใช้เอง
        file_bytes = uploaded.getvalue()  # ไม่ทำ .read() ซ้ำหลายรอบ
        if not is_jpeg_or_png(file_bytes):
            st.error("ไฟล์ที่อัปโหลดไม่ใช่ JPEG/PNG จริง (header ไม่ตรง)")
            st.stop()

        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        image_np = np.array(image)
        st.info("Running YOLO object detection ...")
        results = model.predict(image_np, conf=0.4)

        result_bgr = results[0].plot()
        result_rgb = result_bgr[:, :, ::-1]
        st.image(result_rgb, caption="YOLO Detection Result", use_container_width=True)

        boxes = results[0].boxes
        class_ids = boxes.cls.cpu().numpy().astype(int)
        class_names = [model.names[i] for i in class_ids]
        lp_label = "LicensePlate"  # ปรับให้ตรงกับ class ที่คุณเทรน
        st.write(f"Number of license plate detected: **{class_names.count(lp_label)}**")

    except UnidentifiedImageError as e:
        st.error(f"ไม่สามารถระบุชนิดรูปภาพได้ (Pillow): {e}")
    except Exception as e:
        st.error(f"ไม่สามารถเปิด/ประมวลผลไฟล์ภาพได้: {e}")
