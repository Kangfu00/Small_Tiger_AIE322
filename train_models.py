import os
import cv2
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# 1. การโหลดและเตรียมข้อมูล
base_path = 'static/datasets'
classes = ['56', '57', '58', '59', '60']

X = []
y = []

print("Loading images...")
for label_idx, class_name in enumerate(classes):
    folder_path = os.path.join(base_path, class_name)
    if not os.path.exists(folder_path):
        continue
        
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        # อ่านภาพเป็น Grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            X.append(img)
            y.append(label_idx) # เก็บ label เป็นตัวเลข 0, 1, 2, 3, 4

# แปลงเป็น Numpy Array และ Normalize (0-1)
X = np.array(X, dtype='float32') / 255.0
y = np.array(y)

print(f"Total images loaded: {len(X)}")

# เตรียมข้อมูล 2 รูปแบบสำหรับโมเดลที่ต่างกัน
# รูปแบบที่ 1 สำหรับ ML ปกติ (Decision Tree, Naive Bayes, Random Forest) ต้องแปลงภาพ 28x28 เป็น 1D array (784)
X_ml = X.reshape(-1, 28 * 28)

# รูปแบบที่ 2 สำหรับ CNN ต้องคงมิติภาพไว้และเพิ่ม channel เข้าไป (28, 28, 1)
X_cnn = X.reshape(-1, 28, 28, 1)

# แบ่งข้อมูล Train 80% / Test 20%
X_ml_train, X_ml_test, y_train, y_test = train_test_split(X_ml, y, test_size=0.2, random_state=42)
X_cnn_train, X_cnn_test, _, _ = train_test_split(X_cnn, y, test_size=0.2, random_state=42)

# ฟังก์ชันสำหรับแสดงผลการประเมิน
def evaluate_model(model_name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"--- {model_name} ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("Confusion Matrix:\n", cm)
    print("-" * 20)
    return acc

results = {}

# 2. สร้างและเทรนโมเดล

# โมเดลที่ 1: Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_ml_train, y_train)
y_pred_dt = dt_model.predict(X_ml_test)
results['Decision_Tree'] = evaluate_model("Decision Tree", y_test, y_pred_dt)

# โมเดลที่ 2: Naive Bayes (Gaussian)
nb_model = GaussianNB()
nb_model.fit(X_ml_train, y_train)
y_pred_nb = nb_model.predict(X_ml_test)
results['Naive_Bayes'] = evaluate_model("Naive Bayes", y_test, y_pred_nb)

# โมเดลที่ 3: Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_ml_train, y_train)
y_pred_rf = rf_model.predict(X_ml_test)
results['Random_Forest'] = evaluate_model("Random Forest", y_test, y_pred_rf)

# โมเดลที่ 4: CNN
print("Training CNN...")
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(5, activation='softmax') # 5 คลาสตามช่วงตัวเลข
])

cnn_model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])

# เทรน CNN ซัก 10 epochs
cnn_model.fit(X_cnn_train, y_train, epochs=10, validation_data=(X_cnn_test, y_test), verbose=0)

# ทำนายด้วย CNN
cnn_probs = cnn_model.predict(X_cnn_test)
y_pred_cnn = np.argmax(cnn_probs, axis=1)
results['CNN'] = evaluate_model("CNN", y_test, y_pred_cnn)

# 3. บันทึกโมเดล

# เซฟโมเดล Machine Learning แบบดั้งเดิม (เลือกตัวที่มักจะเสถียรสุด เช่น Random Forest)
joblib.dump(rf_model, 'random_forest_model.pkl')
print("Saved Random Forest model to 'random_forest_model.pkl'")

# เซฟโมเดล Deep Learning (CNN)
cnn_model.save('cnn_model.h5')
print("Saved CNN model to 'cnn_model.h5'")

print("การเทรนเสร็จสมบูรณ์!")