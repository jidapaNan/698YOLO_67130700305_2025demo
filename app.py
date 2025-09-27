import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.title("YOLO Image Detection App :)")

@st.cache_resource
def load_model():
    return YOLO("best.pt")   # ใส่ path ให้ถูกต้อง
model = load_model()

uploaded = st.file_uploader(
    "Upload an image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"]
)

if uploaded is not None:
    try:
        # เปิดไฟล์ภาพตรง ๆ ด้วย PIL
        image = Image.open(uploaded).convert("RGB")

        # แสดงภาพต้นฉบับ
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # แปลงเป็น numpy array เพื่อส่งเข้า YOLO
        image_np = np.array(image)

        st.info("Running YOLO object detection ...")
        results = model.predict(image_np, conf=0.4)

        # วาดผลลัพธ์บนภาพ
        result_bgr = results[0].plot()
        result_rgb = result_bgr[:, :, ::-1]  # BGR → RGB
        st.image(result_rgb, caption="YOLO Detection Result", use_container_width=True)
        st.success("Detection completed!")

        # ดึงผลลัพธ์กล่อง
        boxes = results[0].boxes
        class_ids = boxes.cls.cpu().numpy().astype(int)
        class_names = [model.names[i] for i in class_ids]

        lp_label = "LicensePlate"   # ชื่อ class ต้องตรงกับที่คุณ train
        lp_count = class_names.count(lp_label)
        st.write(f"Number of license plate detected: **{lp_count}**")

    except Exception as e:
        st.error(f"ไม่สามารถเปิด/ประมวลผลไฟล์ภาพได้: {e}")
