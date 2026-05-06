import os
import base64
import time
from flask import Flask, render_template, request, jsonify
from PIL import Image
import io

app = Flask(__name__)

# กำหนดเส้นทางหลักของโฟลเดอร์เก็บข้อมูล
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'datasets')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_canvas():
    try:
        data = request.get_json()
        image_data = data['image'] # รับ Base64 String
        label = data['label']      # รับเลขคลาส (56, 57, 58, 59, 60)

        # 1. จัดการกับ Base64 String (ตัดส่วน prefix ออก)
        # "data:image/png;base64,iVBORw0KG..." -> "iVBORw0KG..."
        header, encoded = image_data.split(",", 1)
        
        # 2. แปลงจาก String กลับเป็นข้อมูล Binary (รูปภาพ)
        image_bytes = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(image_bytes))

        # 3. Preprocessing: แปลงเป็น Grayscale และ Resize เป็น 28x28 (ตามมาตรฐาน CNN)
        # หมายเหตุ: เราวาดบน 280x280 การย่อลงมาจะช่วยให้เส้นดูคมชัดและข้อมูลเล็กลง
        image = image.convert('L') # แปลงเป็นขาวดำ (Grayscale)
        image = image.resize((28, 28), Image.LANCZOS)

        # 4. ตั้งชื่อไฟล์ด้วย Timestamp เพื่อไม่ให้ชื่อซ้ำ
        filename = f"{label}_{int(time.time() * 1000)}.png"
        
        # 5. กำหนดเส้นทางที่จะเซฟ (ไปยังโฟลเดอร์คลาสที่สร้างไว้)
        save_path = os.path.join(DATASET_DIR, label, filename)

        # 6. บันทึกไฟล์
        image.save(save_path)

        return jsonify({"message": "บันทึกสำเร็จ", "filename": filename}), 200

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)