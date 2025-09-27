import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import io

st.title("YOLO Image Detection App :)")

# โหลดโมเดลครั้งเดียวด้วย cache (เร็วและประหยัดทรัพยากร)
@st.cache_resource
def load_model():
    return YOLO("best.pt")   # ใส่ path ให้ถูกกับไฟล์จริงใน repo
model = load_model()

uploaded = st.file_uploader(
    "Upload an image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"]
)

if uploaded is not None:
    try:
        # อ่านเป็น bytes แล้วค่อยเปิดด้วย PIL (กัน pointer เพี้ยน/อ่านซ้ำ)
        img_bytes = uploaded.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # แสดงภาพต้นฉบับ (ส่งเป็น PIL/bytes ก็ได้)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # แปลงเป็น numpy array เพื่อส่งเข้า YOLO
        image_np = np.array(image)

        st.info("Running YOLO object detection ...")
        results = model.predict(image_np, conf=0.4)

        # วาดผลลัพธ์บนภาพ
        result_bgr = results[0].plot()      # BGR (จาก ultralytics)
        result_rgb = result_bgr[:, :, ::-1] # แปลงเป็น RGB เพื่อแสดงบน Streamlit
        st.image(result_rgb, caption="YOLO Detection Result", use_container_width=True)
        st.success("Detection completed!")

        # ดึงผลกล่องและนับคลาสที่ต้องการ
        boxes = results[0].boxes
        class_ids = boxes.cls.cpu().numpy().astype(int)
        class_names = [model.names[i] for i in class_ids]

        # เปลี่ยนชื่อให้ตรงกับที่คุณ train จริง ๆ เช่น 'LicensePlate' หรือ 'license_plate'
        lp_label = "LicensePlate"
        lp_count = sum(1 for n in class_names if n == lp_label)
        st.write(f"Number of license plate detected: **{lp_count}**")

    except Exception as e:
        st.error(f"ไม่สามารถเปิด/ประมวลผลไฟล์ภาพได้: {e}")
