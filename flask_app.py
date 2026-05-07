import os
import base64
import time
from flask import Flask, render_template, request, jsonify
from PIL import Image
import io

app = Flask(__name__)

# กำหนดเส้นทางหลักให้ถูกต้อง
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# แนะนำให้เก็บ datasets ไว้ใน static เพื่อให้เรียกดูรูปผ่านหน้าเว็บได้ง่ายในอนาคต
DATASET_DIR = os.path.join(BASE_DIR, 'static', 'datasets')

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

        # แก้ไขปัญหาพื้นหลังโปร่งใส (ตามที่เคยแนะนำ)
        # สร้างภาพพื้นหลังสีขาวขนาด 280x280 ก่อนนำรูปที่วาดมาวาง
        background = Image.new("RGB", image.size, (255, 255, 255))
        if image.mode == 'RGBA':
            background.paste(image, mask=image.split()[3]) # ใช้ Alpha channel เป็น mask
        else:
            background.paste(image)
        
        # Preprocessing
        image = background.convert('L') 
        image = image.resize((28, 28), Image.LANCZOS)

        # --- จุดแก้ไขสำคัญ: สร้างโฟลเดอร์ถ้ายังไม่มี ---
        target_dir = os.path.join(DATASET_DIR, label)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)
        # ---------------------------------------

        filename = f"{label}_{int(time.time() * 1000)}.png"
        save_path = os.path.join(target_dir, filename)
        image.save(save_path)

        return jsonify({"message": "บันทึกสำเร็จ", "filename": filename}), 200

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

# เพิ่ม Route /predict เพื่อไม่ให้หน้าเว็บค้างเวลาเผลอกดปุ่ม Predict
@app.route('/predict', methods=['POST'])
def predict():
    # ส่วนนี้รอคุณใส่ Model AI ในอนาคต
    return jsonify({"prediction": "ระบบทำนายยังไม่ติดตั้ง"})

if __name__ == '__main__':
    # รันบนเครื่องตัวเองใช้ debug=True เพื่อดู error อย่างละเอียด
    app.run(debug=True, port=5000)