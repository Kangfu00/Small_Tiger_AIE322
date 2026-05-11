import os
import cv2
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ฟังก์ชันเลื่อนภาพ (Data Augmentation)
def shift_image(image, dx, dy):
    rows, cols = image.shape
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(image, M, (cols, rows), borderValue=255)
    return shifted

base_path = 'static/datasets'
classes = ['56', '57', '58', '59', '60']
X_raw, y_raw = [], []

print("1. Loading original images...")
for class_name in classes:
    folder_path = os.path.join(base_path, class_name)
    if not os.path.exists(folder_path): continue
        
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            resized = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
            X_raw.append(resized)
            y_raw.append(class_name) 

# แบ่ง Train/Test
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)

print("2. Augmenting Training Data (x5 multiplier)...")
X_train_aug, y_train_aug = [], []
for img, label in zip(X_train_raw, y_train_raw):
    # 1. ใส่รูปดั้งเดิม
    X_train_aug.append(img)
    y_train_aug.append(label)
    
    # 2. ใส่รูปที่เลื่อนตำแหน่ง (ซ้าย, ขวา, บน, ล่าง)
    for dx, dy in ((2,0), (-2,0), (0,2), (0,-2)):
        X_train_aug.append(shift_image(img, dx, dy))
        # --- แก้ไขตรงนี้: เพิ่ม Label ให้กับรูปที่ถูกเลื่อนด้วย! ---
        y_train_aug.append(label) 

# แปลงเป็น NumPy Array และ Normalize (0-1)
X_train = np.array(X_train_aug, dtype='float32').reshape(-1, 28 * 28) / 255.0
y_train = np.array(y_train_aug)

X_test = np.array(X_test_raw, dtype='float32').reshape(-1, 28 * 28) / 255.0
y_test = np.array(y_test_raw)

print(f"Total training samples: {len(X_train)} | Total labels: {len(y_train)}")

print("3. Training Robust Random Forest (300 trees)...")
rf_model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

print("4. Evaluating Model...")
y_pred = rf_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"=============================")
print(f"FINAL ACCURACY: {acc:.4f} ({(acc*100):.2f}%)")
print(f"=============================")
print(classification_report(y_test, y_pred))

# บันทึกโมเดล
os.makedirs('models', exist_ok=True)
joblib.dump(rf_model, 'models/random_forest_model.pkl')
print("Saved perfectly to 'models/random_forest_model.pkl'")