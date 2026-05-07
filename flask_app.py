import os
import base64
import time
from flask import Flask, render_template, request, jsonify
from PIL import Image
import io
from werkzeug.utils import secure_filename

app = Flask(__name__)

# --- การตั้งค่าเส้นทาง ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'static', 'datasets')
MODEL_DIR = os.path.join(BASE_DIR, 'models') # โฟลเดอร์เก็บโมเดล

# สร้างโฟลเดอร์ที่จำเป็นหากยังไม่มี
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_canvas():
    try:
        data = request.get_json()
        image_data = data['image'] 
        label = data['label']      

        header, encoded = image_data.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(image_bytes))

        # แก้ไขปัญหาพื้นหลังโปร่งใส
        background = Image.new("RGB", image.size, (255, 255, 255))
        if image.mode == 'RGBA':
            background.paste(image, mask=image.split()[3])
        else:
            background.paste(image)
        
        # Preprocessing (28x28 grayscale)
        image = background.convert('L') 
        image = image.resize((28, 28), Image.LANCZOS)

        target_dir = os.path.join(DATASET_DIR, label)
        os.makedirs(target_dir, exist_ok=True)

        filename = f"{label}_{int(time.time() * 1000)}.png"
        save_path = os.path.join(target_dir, filename)
        image.save(save_path)

        return jsonify({"message": "บันทึกข้อมูลสำเร็จ"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- หน้าสำหรับ Admin อัปโหลดโมเดล ---
@app.route('/admin_upload_model', methods=['POST'])
def admin_upload_model():
    try:
        if 'model_file' not in request.files:
            return jsonify({"error": "ไม่มีไฟล์ถูกส่งมา"}), 400
        
        file = request.files['model_file']
        if file.filename == '':
            return jsonify({"error": "ไม่ได้เลือกไฟล์"}), 400

        # ตรวจสอบรหัสผ่านเบื้องต้น (เพื่อความปลอดภัย)
        admin_pass = request.form.get('admin_pass')
        if admin_pass != "1234": # คุณสามารถเปลี่ยนรหัสตรงนี้ได้
            return jsonify({"error": "รหัสผ่านไม่ถูกต้อง"}), 403

        if file:
            # ใช้ชื่อมาตรฐานเพื่อให้โมเดลเรียกใช้ง่าย เช่น model_latest.h5
            ext = os.path.splitext(file.filename)[1]
            filename = "model_latest" + ext
            file.save(os.path.join(MODEL_DIR, filename))
            return jsonify({"message": f"อัปโหลดโมเดล {filename} สำเร็จแล้ว!"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    # ในอนาคตคุณจะใช้โค้ดเรียกไฟล์จาก models/model_latest.h5 มาทายผล
    return jsonify({"prediction": "โมเดลติดตั้งแล้ว (รอการเชื่อมต่อ logic)"})

import shutil # เพิ่มสำหรับบีบอัดไฟล์
from flask import send_file # เพิ่มสำหรับส่งไฟล์ให้ดาวน์โหลด

# --- เพิ่ม Route นี้ลงใน flask_app.py ---

@app.route('/admin_download_datasets', methods=['POST'])
def admin_download_datasets():
    try:
        # ตรวจสอบรหัสผ่าน (ใช้ตัวเดียวกับการอัปโหลดโมเดล)
        admin_pass = request.form.get('admin_pass')
        if admin_pass != "1234": 
            return jsonify({"error": "รหัสผ่านไม่ถูกต้อง"}), 403

        # กำหนดชื่อไฟล์ zip ที่จะสร้าง
        zip_filename = "thai_digits_datasets"
        zip_path = os.path.join(BASE_DIR, zip_filename)

        # ทำการ Zip โฟลเดอร์ datasets (static/datasets)
        # shutil.make_archive จะสร้างไฟล์ .zip ให้อัตโนมัติ
        shutil.make_archive(zip_path, 'zip', DATASET_DIR)

        # ส่งไฟล์ให้ผู้ใช้ดาวน์โหลด
        return send_file(f"{zip_path}.zip", as_attachment=True)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# --- เพิ่ม Route นี้ลงใน flask_app.py ---

@app.route('/admin_stats', methods=['GET'])
def get_stats():
    try:
        stats = {}
        # ลิสต์รายชื่อโฟลเดอร์ใน DATASET_DIR (เช่น 56, 57, ...)
        if os.path.exists(DATASET_DIR):
            for label in os.listdir(DATASET_DIR):
                label_path = os.path.join(DATASET_DIR, label)
                if os.path.isdir(label_path):
                    # นับเฉพาะไฟล์ที่เป็น .png
                    files = [f for f in os.listdir(label_path) if f.endswith('.png')]
                    stats[label] = len(files)
        
        return jsonify(stats), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)