import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
import joblib
import tensorflow as tf

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="ทดสอบโมเดลเลขไทย ๕๖-๖๐", layout="wide")
st.title("🖌️ ทดสอบโมเดลทำนายลายมือเลขไทย (๕๖ - ๖๐)")

# รายชื่อคลาสตามที่เราเทรนไว้ (0=56, 1=57, ...)
CLASS_NAMES = ['๕๖', '๕๗', '๕๘', '๕๙', '๖๐']

# 1. ฟังก์ชันโหลดโมเดล (ใช้ @st.cache_resource เพื่อไม่ให้โหลดใหม่ทุกครั้งที่วาด)
@st.cache_resource
def load_models():
    try:
        rf = joblib.load('random_forest_model.pkl')
        cnn = tf.keras.models.load_model('cnn_model.h5')
        return rf, cnn
    except Exception as e:
        st.error(f"ไม่พบไฟล์โมเดล กรุณาตรวจสอบว่าเทรนเสร็จและมีไฟล์ .pkl, .h5 อยู่ในโฟลเดอร์เดียวกัน: {e}")
        return None, None

rf_model, cnn_model = load_models()

# 2. ฟังก์ชันเตรียมข้อมูลภาพให้เหมือนตอนเทรน
def preprocess_image(img_data):
    # img_data ที่ได้จาก canvas เป็น RGBA (4 channels)
    # แปลงเป็น Grayscale 
    gray = cv2.cvtColor(img_data, cv2.COLOR_RGBA2GRAY)
    
    # ปรับขนาดเป็น 28x28 พิกเซล
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Normalize ให้อยู่ในช่วง 0-1
    normalized = resized.astype('float32') / 255.0
    
    # เตรียม shape สำหรับส่งให้โมเดล
    # สำหรับ Machine Learning (1D array)
    img_ml = normalized.reshape(1, 28 * 28)
    # สำหรับ CNN (3D array with batch size 1)
    img_cnn = normalized.reshape(1, 28, 28, 1)
    
    return img_ml, img_cnn

# แบ่งหน้าจอเป็น 2 คอลัมน์ (ซ้ายวาดรูป, ขวาแสดงผล)
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("วาดเลขไทยที่นี่")
    # สร้างกระดานวาดรูป
    canvas_result = st_canvas(
        fill_color="white",  
        stroke_width=16,     # ขนาดหัวปากกา (ปรับให้ใกล้เคียงภาพที่เทรน)
        stroke_color="black", # สีปากกา
        background_color="white",
        width=280,           # ขนาดกระดาน (เป็นสัดส่วนของ 28x28)
        height=280,
        drawing_mode="freedraw",
        key="canvas",
    )

with col2:
    st.subheader("ผลการทำนาย")
    # ถ้ายูสเซอร์วาดรูปและมีข้อมูลภาพ
    if canvas_result.image_data is not None:
        # เช็คว่ากระดานไม่ได้ว่างเปล่า (มีคนวาดจริงๆ)
        if np.any(canvas_result.image_data[:, :, 3] > 0): 
            img_ml, img_cnn = preprocess_image(canvas_result.image_data)
            
            if rf_model and cnn_model:
                # ทำนายด้วย Random Forest
                rf_pred_idx = rf_model.predict(img_ml)[0]
                rf_result = CLASS_NAMES[rf_pred_idx]
                
                # ทำนายด้วย CNN
                cnn_probs = cnn_model.predict(img_cnn, verbose=0)
                cnn_pred_idx = np.argmax(cnn_probs, axis=1)[0]
                cnn_confidence = np.max(cnn_probs) * 100
                cnn_result = CLASS_NAMES[cnn_pred_idx]
                
                # แสดงผล
                st.success(f"**Random Forest ทายว่า:** {rf_result}")
                st.info(f"**CNN ทายว่า:** {cnn_result} (ความมั่นใจ: {cnn_confidence:.2f}%)")
        else:
            st.write("รอรับภาพวาด...")