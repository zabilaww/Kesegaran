# =========================================================
# TRAINING FINAL
# MobileNetV3 Small + AI Warna + AI Tekstur + SVM
# =========================================================

import os
import cv2
import joblib
import numpy as np

from tqdm import tqdm

from PIL import Image

from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

from tensorflow.keras.applications import MobileNetV3Small

from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

from tensorflow.keras.layers import GlobalAveragePooling2D

from tensorflow.keras.models import Model

# =========================================================
# DATASET
# =========================================================

DATASET_PATH = r"C:\Bila\testing\dataset\dataset_split"

# =========================================================
# MODEL FOLDER
# =========================================================

MODEL_PATH = r"C:\Bila\testing\model"

os.makedirs(
    MODEL_PATH,
    exist_ok=True
)

# =========================================================
# IMAGE SIZE
# =========================================================

IMG_SIZE = 224

# =========================================================
# LOAD MOBILENETV3
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

print("Model berhasil dimuat")

# =========================================================
# COLOR FEATURE
# =========================================================

def extract_color_features(image):

    # =============================================
    # RGB MEAN
    # =============================================

    mean_r = np.mean(image[:,:,0])

    mean_g = np.mean(image[:,:,1])

    mean_b = np.mean(image[:,:,2])

    # =============================================
    # HSV
    # =============================================

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2HSV
    )

    h_mean = np.mean(hsv[:,:,0])

    s_mean = np.mean(hsv[:,:,1])

    v_mean = np.mean(hsv[:,:,2])

    # =============================================
    # BRIGHTNESS
    # =============================================

    brightness = np.mean(

        cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY
        )

    )

    # =============================================
    # REDNESS
    # =============================================

    redness = mean_r - (
        (mean_g + mean_b) / 2
    )

    return np.array([

        mean_r,
        mean_g,
        mean_b,

        h_mean,
        s_mean,
        v_mean,

        brightness,

        redness

    ])

# =========================================================
# TEXTURE FEATURE
# =========================================================

def extract_texture_features(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    # =============================================
    # LAPLACIAN VARIANCE
    # =============================================

    laplacian_var = cv2.Laplacian(

        gray,

        cv2.CV_64F

    ).var()

    # =============================================
    # EDGE DENSITY
    # =============================================

    edges = cv2.Canny(
        gray,
        100,
        200
    )

    edge_density = np.sum(
        edges > 0
    )

    # =============================================
    # CONTRAST
    # =============================================

    contrast = gray.std()

    return np.array([

        laplacian_var,

        edge_density,

        contrast

    ])

# =========================================================
# CNN FEATURE
# =========================================================

def extract_cnn_feature(image):

    image = image.astype(
        np.float32
    ) / 255.0

    image = preprocess_input(
        image
    )

    image = np.expand_dims(
        image,
        axis=0
    )

    feature = feature_extractor.predict(

        image,

        verbose=0

    )

    return feature.flatten()

# =========================================================
# FINAL FEATURE
# =========================================================

def extract_final_feature(image):

    cnn_feature = extract_cnn_feature(
        image
    )

    color_feature = extract_color_features(
        image
    )

    texture_feature = extract_texture_features(
        image
    )

    final_feature = np.concatenate([

        cnn_feature,

        color_feature,

        texture_feature

    ])

    return final_feature

# =========================================================
# LOAD DATASET
# =========================================================

def load_dataset(split_name):

    data = []

    labels = []

    split_path = os.path.join(
        DATASET_PATH,
        split_name
    )

    label_map = {

        "segar": 1,

        "tidak_segar": 0

    }

    for label_name, label_value in label_map.items():

        folder = os.path.join(
            split_path,
            label_name
        )

        if not os.path.exists(folder):
            continue

        files = [

            f for f in os.listdir(folder)

            if f.lower().endswith((
                '.jpg',
                '.jpeg',
                '.png',
                '.webp'
            ))

        ]

        print(f"\n{split_name.upper()} - {label_name}")
        print(f"Jumlah data : {len(files)}")

        for file in tqdm(files):

            file_path = os.path.join(
                folder,
                file
            )

            try:

                # =====================================
                # LOAD IMAGE
                # =====================================

                image = Image.open(
                    file_path
                ).convert("RGB")

                image = image.resize(
                    (IMG_SIZE, IMG_SIZE)
                )

                image = np.array(image)

                # =====================================
                # FINAL FEATURE
                # =====================================

                feature = extract_final_feature(
                    image
                )

                data.append(feature)

                labels.append(label_value)

            except Exception as e:

                print(f"Error : {file}")
                print(e)

    return np.array(data), np.array(labels)

# =========================================================
# MEMBACA DATASET
# =========================================================

print("\n===================================")
print("MEMBACA DATASET")
print("===================================\n")

X_train, y_train = load_dataset("train")

X_valid, y_valid = load_dataset("valid")

X_test, y_test = load_dataset("test")

# =========================================================
# INFORMASI DATASET
# =========================================================

print("\n===================================")
print("INFORMASI DATASET")
print("===================================\n")

print(f"Train : {X_train.shape}")
print(f"Valid : {X_valid.shape}")
print(f"Test  : {X_test.shape}")

# =========================================================
# FEATURE SCALING
# =========================================================

print("\n===================================")
print("FEATURE SCALING")
print("===================================\n")

scaler = StandardScaler()

X_train = scaler.fit_transform(
    X_train
)

X_valid = scaler.transform(
    X_valid
)

X_test = scaler.transform(
    X_test
)

# =========================================================
# TRAINING SVM
# =========================================================

print("\n===================================")
print("TRAINING SMART AI")
print("===================================\n")

svm_model = SVC(

    kernel='rbf',

    C=10,

    gamma='scale',

    probability=True,

    class_weight='balanced'

)

svm_model.fit(
    X_train,
    y_train
)

print("Training selesai")

# =========================================================
# VALIDASI
# =========================================================

print("\n===================================")
print("VALIDASI")
print("===================================\n")

valid_pred = svm_model.predict(
    X_valid
)

valid_acc = accuracy_score(
    y_valid,
    valid_pred
)

print(f"Akurasi Validasi : {valid_acc*100:.2f}%")

# =========================================================
# TESTING
# =========================================================

print("\n===================================")
print("TESTING")
print("===================================\n")

test_pred = svm_model.predict(
    X_test
)

test_acc = accuracy_score(
    y_test,
    test_pred
)

print(f"Akurasi Testing : {test_acc*100:.2f}%")

# =========================================================
# CONFUSION MATRIX
# =========================================================

print("\n===================================")
print("CONFUSION MATRIX")
print("===================================\n")

cm = confusion_matrix(
    y_test,
    test_pred
)

print(cm)

# =========================================================
# CLASSIFICATION REPORT
# =========================================================

print("\n===================================")
print("CLASSIFICATION REPORT")
print("===================================\n")

print(

    classification_report(

        y_test,
        test_pred,

        target_names=[

            "Tidak Segar",

            "Segar"

        ]

    )

)

# =========================================================
# SAVE MODEL
# =========================================================

joblib.dump(

    svm_model,

    os.path.join(
        MODEL_PATH,
        "svm_mobilenetv3.pkl"
    )

)

joblib.dump(

    scaler,

    os.path.join(
        MODEL_PATH,
        "scaler.pkl"
    )

)

print("\n===================================")
print("MODEL BERHASIL DISIMPAN")
print("===================================\n")

print("Lokasi model :")

print(

    os.path.join(
        MODEL_PATH,
        "svm_mobilenetv3.pkl"
    )

)