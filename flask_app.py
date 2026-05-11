import os
import base64
import time
import io
import numpy as np
import shutil
import joblib
import cv2
from flask import Flask, render_template, request, jsonify, send_file
from PIL import Image

app = Flask(__name__)

# --- การตั้งค่าเส้นทาง ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'static', 'datasets')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
# ใช้ชื่อไฟล์โมเดล Random Forest ที่คุณมี
MODEL_PATH = os.path.join(MODEL_DIR, 'random_forest_model_v1.joblib')

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

model = None
def load_my_model():
    global model
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            print('✅ โหลด Random Forest Model สำเร็จ!')
        except Exception as e:
            print(f'❌ โหลดโมเดลไม่สำเร็จ: {e}')

load_my_model()

def center_and_resize(img_gray):
    # Invert และ Threshold เพื่อหาตัวเลข (ลายเส้นขาวบนพื้นดำ)
    _, thresh = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return cv2.resize(thresh, (28, 28), interpolation=cv2.INTER_AREA)
    
    x, y, w, h = cv2.boundingRect(coords)
    cropped = thresh[y:y+h, x:x+w]
    max_side = max(w, h)
    square = np.zeros((max_side, max_side), dtype=np.uint8)
    off_x = (max_side - w) // 2
    off_y = (max_side - h) // 2
    square[off_y:off_y+h, off_x:off_x+w] = cropped
    
    return cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global model
    try:
        if model is None:
            return jsonify({'error': 'ไม่พบไฟล์โมเดลในระบบ'}), 404

        data = request.get_json()
        image_data = data['image']
        
        header, encoded = image_data.split(',', 1)
        image_bytes = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(image_bytes))

        background = Image.new('RGB', image.size, (255, 255, 255))
        if image.mode == 'RGBA':
            background.paste(image, mask=image.split()[3])
        else:
            background.paste(image)
        
        img_gray = np.array(background.convert('L'))
        processed_img = center_and_resize(img_gray)
        img_array = processed_img.reshape(1, -1).astype('float32') / 255.0

        prediction = model.predict(img_array)[0]
        try:
            probs = model.predict_proba(img_array)
            confidence = float(np.max(probs))
        except:
            confidence = 0.0

        return jsonify({
            'prediction': str(prediction),
            'confidence': f'{confidence * 100:.2f}%'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_canvas():
    try:
        data = request.get_json()
        image_data = data['image']
        label = data['label']
        header, encoded = image_data.split(',', 1)
        image_bytes = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(image_bytes))
        background = Image.new('RGB', image.size, (255, 255, 255))
        if image.mode == 'RGBA': background.paste(image, mask=image.split()[3])
        else: background.paste(image)
        image_to_save = background.convert('L').resize((28, 28), Image.LANCZOS)
        target_dir = os.path.join(DATASET_DIR, label)
        os.makedirs(target_dir, exist_ok=True)
        filename = f'{label}_{int(time.time() * 1000)}.png'
        image_to_save.save(os.path.join(target_dir, filename))
        return jsonify({'message': 'บันทึกสำเร็จ'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/admin_upload_model', methods=['POST'])
def admin_upload_model():
    try:
        if 'model_file' not in request.files: return jsonify({'error': 'ไม่มีไฟล์'}), 400
        file = request.files['model_file']
        if request.form.get('admin_pass') != '1234': return jsonify({'error': 'รหัสผ่านไม่ถูกต้อง'}), 403
        if file and file.filename != '':
            file.save(MODEL_PATH)
            load_my_model()
            return jsonify({'message': 'อัปโหลดโมเดลสำเร็จ!'}), 200
        return jsonify({'error': 'ไฟล์ไม่ถูกต้อง'}), 400
    except Exception as e: return jsonify({'error': str(e)}), 500

@app.route('/admin_download_datasets', methods=['POST'])
def admin_download_datasets():
    try:
        if request.form.get('admin_pass') != '1234': return jsonify({'error': 'รหัสผ่านไม่ถูกต้อง'}), 403
        zip_name = 'thai_digits_datasets'
        zip_full_path = os.path.join(BASE_DIR, zip_name)
        shutil.make_archive(zip_full_path, 'zip', DATASET_DIR)
        return send_file(f'{zip_full_path}.zip', as_attachment=True)
    except Exception as e: return jsonify({'error': str(e)}), 500

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
    except Exception as e: return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)