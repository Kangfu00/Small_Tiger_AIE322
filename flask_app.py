import os
import base64
import time
import io
import numpy as np
import shutil
import joblib  # ใช้ joblib แทน tensorflow
from flask import Flask, render_template, request, jsonify, send_file
from PIL import Image
import cv2

app = Flask(__name__)

# --- การตั้งค่าเส้นทาง ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'static', 'datasets')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
# เปลี่ยนนามสกุลไฟล์โมเดลเป็น .joblib
MODEL_PATH = os.path.join(MODEL_DIR, 'random_forest_model_v_1_0.joblib')

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def center_and_resize(img_gray):
    _, thresh = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return cv2.resize(img_gray, (28, 28), interpolation=cv2.INTER_AREA)
    
    x, y, w, h = cv2.boundingRect(coords)
    cropped = thresh[y:y+h, x:x+w]
    max_side = max(w, h)
    square = np.zeros((max_side, max_side), dtype=np.uint8)
    
    start_x = (max_side - w) // 2
    start_y = (max_side - h) // 2
    square[start_y:start_y+h, start_x:start_x+w] = cropped
    
    resized = cv2.resize(square, (20, 20), interpolation=cv2.INTER_AREA)
    padded = np.pad(resized, ((4,4), (4,4)), mode='constant', constant_values=0)
    final_img = cv2.bitwise_not(padded)
    return final_img

model = None
def load_my_model():
    global model
    if os.path.exists(MODEL_PATH):
        try:
            # โหลด Random Forest Model
            model = joblib.load(MODEL_PATH)
            print("✅ โหลด Random Forest Model สำเร็จ!")
        except Exception as e:
            print(f"❌ โหลดโมเดลไม่สำเร็จ: {e}")

load_my_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global model
    try:
        if model is None:
            return jsonify({"error": "ไม่พบโมเดลในระบบ"}), 404

        data = request.get_json()
        image_data = data['image']
        
        header, encoded = image_data.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(image_bytes))

        # 1. จัดการพื้นหลัง
        background = Image.new("RGB", image.size, (255, 255, 255))
        if image.mode == 'RGBA':
            background.paste(image, mask=image.split()[3])
        else:
            background.paste(image)
        
        # 2. แปลงเป็น Grayscale
        gray_img = cv2.cvtColor(np.array(background), cv2.COLOR_RGB2GRAY)

        # 3. *** แก้ไขจุดนี้: ใช้ฟังก์ชันจัดกึ่งกลางที่เขียนไว้ ***
        processed_img = center_and_resize(gray_img) 
        # ฟังก์ชันนี้จะทำการ Invert สีให้เสร็จสรรพ (ตัวเลขขาว พื้นดำ) ตามโค้ดที่คุณเขียนไว้ในฟังก์ชัน

        # 4. ปรับขนาดและเตรียม Array (Flatten)
        img_array = processed_img.reshape(1, -1).astype('float32') / 255.0

        # 5. ทำนายผล
        prediction = model.predict(img_array)[0]
        
        try:
            probabilities = model.predict_proba(img_array)
            confidence = float(np.max(probabilities))
        except:
            confidence = 1.0

        return jsonify({
            "prediction": str(prediction),
            "confidence": f"{confidence * 100:.2f}%"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- หน้าสำหรับการเก็บข้อมูล (Collect Data) ---
@app.route('/upload', methods=['POST'])
def upload_canvas():
    try:
        data = request.get_json()
        image_data = data['image'] 
        label = data['label']      

        header, encoded = image_data.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(image_bytes))

        background = Image.new("RGB", image.size, (255, 255, 255))
        if image.mode == 'RGBA':
            background.paste(image, mask=image.split()[3])
        else:
            background.paste(image)
        
        image = background.convert('L').resize((28, 28), Image.LANCZOS)
        target_dir = os.path.join(DATASET_DIR, label)
        os.makedirs(target_dir, exist_ok=True)

        filename = f"{label}_{int(time.time() * 1000)}.png"
        image.save(os.path.join(target_dir, filename))
        return jsonify({"message": "บันทึกสำเร็จ"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- ระบบ Admin ---
@app.route('/admin_upload_model', methods=['POST'])
def admin_upload_model():
    try:
        if 'model_file' not in request.files:
            return jsonify({"error": "ไม่มีไฟล์ถูกส่งมา"}), 400
        
        file = request.files['model_file']
        admin_pass = request.form.get('admin_pass')
        if admin_pass != "1234":
            return jsonify({"error": "รหัสผ่านไม่ถูกต้อง"}), 403

        if file and file.filename != '':
            file.save(MODEL_PATH)
            load_my_model() 
            return jsonify({"message": "อัปโหลดและเปิดใช้งานโมเดลใหม่สำเร็จ!"}), 200
        return jsonify({"error": "ไฟล์ไม่ถูกต้อง"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# แก้ไขจุดนี้: ลบ @app.download_handler ออก
@app.route('/admin_download_datasets', methods=['POST'])
def admin_download_datasets():
    try:
        admin_pass = request.form.get('admin_pass')
        if admin_pass != "1234": 
            return jsonify({"error": "รหัสผ่านไม่ถูกต้อง"}), 403

        zip_name = "thai_digits_datasets"
        zip_full_path = os.path.join(BASE_DIR, zip_name)
        
        # สร้าง Zip
        shutil.make_archive(zip_full_path, 'zip', DATASET_DIR)
        
        return send_file(f"{zip_full_path}.zip", as_attachment=True)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/admin_stats', methods=['GET'])
def get_stats():
    try:
        stats = {}
        if os.path.exists(DATASET_DIR):
            for label in sorted(os.listdir(DATASET_DIR)):
                label_path = os.path.join(DATASET_DIR, label)
                if os.path.isdir(label_path):
                    files = [f for f in os.listdir(label_path) if f.endswith('.png')]
                    stats[label] = len(files)
        return jsonify(stats), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)