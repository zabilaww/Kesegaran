# =========================================================
# HASIL KLASIFIKASI SVM DENGAN NAMA FILE ASLI
# =========================================================

# =========================================================
# IMPORT LIBRARY
# =========================================================

import os
import cv2
import joblib
import numpy as np
import pandas as pd

from tqdm import tqdm

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)

from tensorflow.keras.applications import MobileNetV3Small

from tensorflow.keras.applications.mobilenet_v3 import (
    preprocess_input
)

from tensorflow.keras.layers import GlobalAveragePooling2D

from tensorflow.keras.models import Model

# =========================================================
# PATH DATASET
# =========================================================

DATASET_PATH = r"C:\Rangga\SKRIPSI\BILA\Website\dataset\dataset_split\test"

# =========================================================
# PATH MODEL
# =========================================================

MODEL_PATH = r"C:\Rangga\SKRIPSI\BILA\Website\model\svm_mobilenetv3.pkl"

SCALER_PATH = r"C:\Rangga\SKRIPSI\BILA\Website\model\scaler.pkl"

# =========================================================
# OUTPUT CSV
# =========================================================

OUTPUT_CSV = r"C:\Rangga\SKRIPSI\BILA\Website\hasil_klasifikasi_testing.csv"

# =========================================================
# IMAGE SIZE
# =========================================================

IMG_SIZE = 224

# =========================================================
# MEMUAT MODEL SVM
# =========================================================

print("\n===================================")
print("MEMUAT MODEL SVM")
print("===================================\n")

svm_model = joblib.load(MODEL_PATH)

scaler = joblib.load(SCALER_PATH)

print("Model berhasil dimuat")

# =========================================================
# MEMUAT MOBILENETV3 SMALL
# =========================================================

print("\n===================================")
print("MEMUAT MOBILENETV3 SMALL")
print("===================================\n")

base_model = MobileNetV3Small(

    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)

)

x = GlobalAveragePooling2D()(base_model.output)

feature_extractor = Model(

    inputs=base_model.input,
    outputs=x

)

print("MobileNetV3 berhasil dimuat")

# =========================================================
# LABEL
# =========================================================

label_map = {

    0: "tidak_segar",
    1: "segar"

}

# =========================================================
# LIST HASIL
# =========================================================

results = []

# =========================================================
# LIST LABEL
# =========================================================

y_true = []
y_pred = []

# =========================================================
# FUNGSI EKSTRAKSI FITUR
# =========================================================

def extract_feature(image_path):

    # ==========================================
    # MEMBACA GAMBAR
    # ==========================================

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Gagal membaca gambar: {image_path}")

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = cv2.resize(
        image,
        (IMG_SIZE, IMG_SIZE)
    )

    # ==========================================
    # COPY UNTUK FITUR TAMBAHAN
    # ==========================================

    image_feature = image.copy()

    # ==========================================
    # PREPROCESS MOBILENET
    # ==========================================

    image = image.astype(np.float32)

    image = preprocess_input(image)

    image = np.expand_dims(
        image,
        axis=0
    )

    # ==========================================
    # EKSTRAKSI FITUR CNN
    # ==========================================

    cnn_feature = feature_extractor.predict(
        image,
        verbose=0
    ).flatten()

    # =====================================================
    # FITUR TAMBAHAN
    # =====================================================

    # Mean RGB
    mean_r = np.mean(image_feature[:, :, 0])
    mean_g = np.mean(image_feature[:, :, 1])
    mean_b = np.mean(image_feature[:, :, 2])

    # Standard Deviation RGB
    std_r = np.std(image_feature[:, :, 0])
    std_g = np.std(image_feature[:, :, 1])
    std_b = np.std(image_feature[:, :, 2])

    # Grayscale
    gray = cv2.cvtColor(
        image_feature,
        cv2.COLOR_RGB2GRAY
    )

    # Brightness
    brightness = np.mean(gray)

    # Contrast
    contrast = np.std(gray)

    # Edge Detection
    edges = cv2.Canny(
        gray,
        100,
        200
    )

    edge_mean = np.mean(edges)

    # Saturation
    hsv = cv2.cvtColor(
        image_feature,
        cv2.COLOR_RGB2HSV
    )

    saturation = np.mean(hsv[:, :, 1])

    # Variance
    variance = np.var(gray)

    # =====================================================
    # TOTAL FITUR TAMBAHAN = 11
    # =====================================================

    additional_feature = np.array([

        mean_r,
        mean_g,
        mean_b,

        std_r,
        std_g,
        std_b,

        brightness,
        contrast,
        edge_mean,
        saturation,

        variance

    ])

    # =====================================================
    # GABUNGKAN FITUR
    # 576 + 11 = 587
    # =====================================================

    final_feature = np.concatenate([

        cnn_feature,
        additional_feature

    ])

    return final_feature

# =========================================================
# MEMBACA DATA TESTING
# =========================================================

print("\n===================================")
print("MEMBACA DATA TESTING")
print("===================================\n")

classes = [

    "segar",
    "tidak_segar"

]

no = 1

for class_name in classes:

    class_folder = os.path.join(
        DATASET_PATH,
        class_name
    )

    files = [

        f for f in os.listdir(class_folder)

        if f.lower().endswith((
            ".jpg",
            ".jpeg",
            ".png"
        ))

    ]

    print(f"{class_name} : {len(files)} citra")

    for file in tqdm(files):

        try:

            image_path = os.path.join(
                class_folder,
                file
            )

            # =====================================
            # EKSTRAKSI FITUR
            # =====================================

            feature = extract_feature(
                image_path
            )

            # =====================================
            # FEATURE SCALING
            # =====================================

            feature = scaler.transform(
                [feature]
            )

            # =====================================
            # PREDIKSI
            # =====================================

            pred = svm_model.predict(
                feature
            )[0]

            pred_label = label_map[pred]

            # =====================================
            # STATUS
            # =====================================

            if pred_label == class_name:
                status = "Benar"
            else:
                status = "Salah"

            # =====================================
            # SIMPAN LABEL
            # =====================================

            y_true.append(class_name)
            y_pred.append(pred_label)

            # =====================================
            # SIMPAN HASIL
            # =====================================

            results.append({

                "No": no,
                "Nama File Gambar": file,
                "Label Asli": class_name,
                "Label Prediksi": pred_label,
                "Status": status

            })

            no += 1

        except Exception as e:

            print(f"\nError : {file}")
            print(e)

# =========================================================
# DATAFRAME
# =========================================================

df = pd.DataFrame(results)

# =========================================================
# SIMPAN CSV
# =========================================================

df.to_csv(

    OUTPUT_CSV,
    index=False

)

# =========================================================
# HITUNG BENAR SALAH
# =========================================================

jumlah_benar = sum(df["Status"] == "Benar")

jumlah_salah = sum(df["Status"] == "Salah")

# =========================================================
# ACCURACY
# =========================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

# =========================================================
# CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=["tidak_segar", "segar"]
)

# =========================================================
# CLASSIFICATION REPORT
# =========================================================

report = classification_report(
    y_true,
    y_pred
)

# =========================================================
# MENAMPILKAN HASIL
# =========================================================

print("\n===================================")
print("HASIL KLASIFIKASI")
print("===================================\n")

print(df)

# =========================================================
# MENAMPILKAN EVALUASI
# =========================================================

print("\n===================================")
print("HASIL EVALUASI")
print("===================================\n")

print(f"Total Data          : {len(df)}")
print(f"Prediksi Benar      : {jumlah_benar}")
print(f"Prediksi Salah      : {jumlah_salah}")
print(f"Accuracy            : {accuracy*100:.2f}%")

# =========================================================
# MENAMPILKAN CONFUSION MATRIX
# =========================================================

print("\n===================================")
print("CONFUSION MATRIX")
print("===================================\n")

print(cm)

# =========================================================
# MENAMPILKAN CLASSIFICATION REPORT
# =========================================================

print("\n===================================")
print("CLASSIFICATION REPORT")
print("===================================\n")

print(report)

# =========================================================
# INFORMASI CSV
# =========================================================

print("\n===================================")
print("CSV BERHASIL DISIMPAN")
print("===================================\n")

print(OUTPUT_CSV)