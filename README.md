# Thai Digits AI ๕๖-๖๐

โครงการพัฒนาระบบทำนายลายมือเขียนตัวเลขไทย (๕๖ - ๖๐) โดยใช้ Machine Learning ด้วย Random Forest Classifier

## ฟีเจอร์หลัก

- **หน้าเก็บข้อมูลและสถิติ**: วาดตัวเลขบน Canvas และบันทึกเป็นข้อมูลสำหรับเทรนโมเดล
- **หนาทดสอบ AI**: วาดตัวเลขและรับผลทำนายแบบ Real-time
- **หน้า Admin**: อัปโหลดโมเดลใหม่และดาวน์โหลด Dataset

## สิ่งที่ต้องติดตั้งก่อน

1. **Python**: ต้องมี Python ติดตั้ง (แนะนำ Python 3.8 ขึ้นไป)

2. **Virtual Environment**: โปรเจกต์นี้ใช้ virtual environment ชื่อ `aie_env`

3. **Dependencies**: ติดตั้งไลบรารีที่จำเป็นโดยรันคำสั่ง:

   ```
   pip install flask numpy opencv-python pillow scikit-learn joblib
   ```

   หรือถ้าอยู่ใน virtual env แล้ว:

   ```
   aie_env\Scripts\activate  # บน Windows
   pip install flask numpy opencv-python pillow scikit-learn joblib
   ```

## วิธีรันโปรแกรม

1. **Activate Virtual Environment**:

   ```
   aie_env\Scripts\activate
   ```

2. **รัน Flask App**:

   ```
   python flask_app.py
   ```

3. **เปิดเบราว์เซอร์**: ไปที่ `http://localhost:5000`

## แนะนำหน้าเว็บ

### หน้าแรก: เก็บข้อมูล & สถิติ

- เลือกตัวเลขที่ต้องการวาด (๕๖ - ๖๐)
- วาดบน Canvas
- กด "บันทึกข้อมูล" เพื่อเพิ่มรูปภาพเข้า Dataset
- ดูสถิติจำนวนรูปภาพในแต่ละคลาส

### หนาทดสอบ AI: ทดสอบ AI

- วาดตัวเลขบน Canvas
- กด "ทำนายผลทันที" เพื่อให้ AI ทำนาย
- ผลทำนายจะแสดงด้านล่าง

### หน้า Admin: ตั้งค่าระบบ

- ใส่รหัสผ่าน (default: 1234)
- ดาวน์โหลด Dataset เป็นไฟล์ ZIP
- อัปโหลดโมเดลใหม่ (.pkl file)

## โครงสร้างโปรเจกต์

- `flask_app.py`: โค้ดหลักของ Flask app
- `train_models.py`: โค้ดสำหรับเทรนโมเดล
- `templates/index.html`: หน้าเว็บหลัก
- `static/css/style.css`: สไตล์ CSS
- `static/js/main.js`: JavaScript สำหรับการทำงานของ Canvas และ API calls
- `static/datasets/`: โฟลเดอร์เก็บ Dataset (56,57,58,59,60)
- `models/`: โฟลเดอร์เก็บโมเดลที่เทรนแล้ว

## การเทรนโมเดล

รัน `python train_models.py` เพื่อเทรนโมเดลใหม่จาก Dataset ใน `static/datasets/`

โมเดลจะถูกบันทึกเป็น `models/random_forest_model.pkl`

## หมายเหตุ

- โมเดลใช้ Random Forest Classifier จาก scikit-learn
- รูปภาพถูก preprocess เป็น 28x28 pixels และ normalize เป็น 0-1
- ใช้ Data Augmentation โดยการเลื่อนภาพเพื่อเพิ่มข้อมูล

สำหรับคำถามเพิ่มเติม ติดต่อผู้พัฒนา.