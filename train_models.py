import os
import cv2
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# --- ฟังก์ชันจัดกึ่งกลางอัตโนมัติ ---
def center_and_resize(img_gray):
    # แปลงให้หมึกเป็นสีขาว (255) พื้นหลังสีดำ (0) เพื่อให้หาง่าย
    _, thresh = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # หาพิกัดของลายเส้น
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return cv2.resize(img_gray, (28, 28), interpolation=cv2.INTER_AREA)
    
    # หา Bounding Box (กรอบสี่เหลี่ยมที่ครอบลายเส้นพอดี)
    x, y, w, h = cv2.boundingRect(coords)
    cropped = thresh[y:y+h, x:x+w]
    
    # ทำกรอบให้เป็นสี่เหลี่ยมจัตุรัส โดยอิงจากด้านที่ยาวที่สุด
    max_side = max(w, h)
    square = np.zeros((max_side, max_side), dtype=np.uint8)
    
    # วางลายเส้นลงไปตรงกลางจัตุรัส
    start_x = (max_side - w) // 2
    start_y = (max_side - h) // 2
    square[start_y:start_y+h, start_x:start_x+w] = cropped
    
    # ย่อขนาดลงมาที่ 20x20 และเติมขอบขาวด้านละ 4 พิกเซลให้กลายเป็น 28x28 (สูตรเดียวกับ MNIST)
    resized = cv2.resize(square, (20, 20), interpolation=cv2.INTER_AREA)
    padded = np.pad(resized, ((4,4), (4,4)), mode='constant', constant_values=0)
    
    # แปลงกลับให้พื้นเป็นสีขาว หมึกสีดำ เหมือนภาพต้นฉบับ
    final_img = cv2.bitwise_not(padded)
    return final_img

# 1. การโหลดและเตรียมข้อมูล
base_path = 'static/datasets'
classes = ['56', '57', '58', '59', '60']

X = []
y = []

print("Loading and centering images...")
for class_name in classes:
    folder_path = os.path.join(base_path, class_name)
    if not os.path.exists(folder_path):
        continue
        
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            # นำรูปผ่านกระบวนการจัดกึ่งกลางก่อนเก็บเข้า Array
            centered_img = center_and_resize(img)
            X.append(centered_img)
            y.append(class_name) 

X = np.array(X, dtype='float32') / 255.0
y = np.array(y) 

print(f"Total images loaded: {len(X)}")

X_ml = X.reshape(-1, 28 * 28)
X_train, X_test, y_train, y_test = train_test_split(X_ml, y, test_size=0.2, random_state=42)

# 2. สร้างและเทรน Random Forest (อัปเกรดเป็น 300 ต้น)
print("Training Random Forest (300 estimators)...")
rf_model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# ฟังก์ชันประเมินผล
y_pred = rf_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred, labels=classes)

print(f"--- Random Forest ---")
print(f"Accuracy:  {acc:.4f}")
print("Confusion Matrix:\n", cm)
print("-" * 20)

# 3. บันทึกโมเดล
joblib.dump(rf_model, 'random_forest_model.pkl')
print("Saved upgraded model to 'random_forest_model.pkl'")