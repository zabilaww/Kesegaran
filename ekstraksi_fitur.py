# =========================================================
# EKSTRAKSI FITUR IKAN KEMBUNG
# MobileNetV3 Small
# =========================================================

import os
import numpy as np
import pandas as pd

from tqdm import tqdm

from PIL import Image

from tensorflow.keras.applications import MobileNetV3Small

from tensorflow.keras.applications.mobilenet_v3 import (
    preprocess_input
)

from tensorflow.keras.layers import GlobalAveragePooling2D

from tensorflow.keras.models import Model

# =========================================================
# DATASET PATH
# =========================================================

DATASET_PATH = r"C:\Rangga\SKRIPSI\BILA\Website\dataset\dataset_split"

# =========================================================
# OUTPUT CSV
# =========================================================

OUTPUT_CSV = r"C:\Rangga\SKRIPSI\BILA\Website\hasil_ekstraksi_fitur.csv"

# =========================================================
# IMAGE SIZE
# =========================================================

IMG_SIZE = 224

# =========================================================
# LOAD MOBILENETV3 SMALL
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
# DATASET SPLIT
# =========================================================

dataset_folders = [

    "train",
    "valid",
    "test"

]

# =========================================================
# MENYIMPAN HASIL
# =========================================================

all_data = []

# =========================================================
# EKSTRAKSI FITUR
# =========================================================

def extract_feature(image_path):

    # =============================================
    # LOAD IMAGE
    # =============================================

    image = Image.open(
        image_path
    ).convert("RGB")

    # =============================================
    # RESIZE
    # =============================================

    image = image.resize(
        (IMG_SIZE, IMG_SIZE)
    )

    # =============================================
    # NUMPY
    # =============================================

    image = np.array(image)

    # =============================================
    # NORMALISASI
    # =============================================

    image = image.astype(
        np.float32
    ) / 255.0

    # =============================================
    # PREPROCESS INPUT
    # =============================================

    image = preprocess_input(
        image
    )

    # =============================================
    # EXPAND DIMENSION
    # =============================================

    image = np.expand_dims(
        image,
        axis=0
    )

    # =============================================
    # FEATURE EXTRACTION
    # =============================================

    feature = feature_extractor.predict(

        image,

        verbose=0

    )

    return feature.flatten()

# =========================================================
# MEMBACA DATASET
# =========================================================

print("\n===================================")
print("MEMBACA DATASET SEGAR")
print("===================================\n")

for dataset_name in dataset_folders:

    # =============================================
    # FOLDER SEGAR
    # =============================================

    segar_folder = os.path.join(

        DATASET_PATH,

        dataset_name,

        "segar"

    )

    if not os.path.exists(segar_folder):
        continue

    # =============================================
    # AMBIL FILE GAMBAR
    # =============================================

    files = [

        f for f in os.listdir(segar_folder)

        if f.lower().endswith((
            ".jpg",
            ".jpeg",
            ".png",
            ".webp"
        ))

    ]

    print(f"{dataset_name.upper()} : {len(files)} citra")

    # =============================================
    # LOOP FILE
    # =============================================

    for file in tqdm(files):

        file_path = os.path.join(
            segar_folder,
            file
        )

        try:

            # =====================================
            # NAMA FILE
            # =====================================

            filename = os.path.splitext(
                file
            )[0]

            parts = filename.split("_")

            # =====================================
            # AMBIL INFORMASI
            # =====================================

            ikan = "-"
            jam = "-"
            bagian = "-"

            for p in parts:

                # ikan01
                if "ikan" in p.lower():

                    ikan = p

                # 07 / 09 / 11
                elif p.isdigit():

                    jam = p

                # mata / insang / tekstur
                elif p.lower() in [

                    "mata",
                    "insang",
                    "tekstur"

                ]:

                    bagian = p

            # =====================================
            # EKSTRAKSI FITUR
            # =====================================

            feature = extract_feature(
                file_path
            )

            # =====================================
            # AMBIL F1 - F50
            # =====================================

            feature_data = {}

            for i in range(50):

                feature_data[f"F{i+1}"] = round(

                    float(feature[i]),

                    6

                )

            # =====================================
            # FITUR TERAKHIR
            # =====================================

            feature_data["Fitur_Terakhir"] = round(

                float(feature[-1]),

                6

            )

            # =====================================
            # SIMPAN DATA
            # =====================================

            row = {

                "Dataset": dataset_name,
                "Ikan": ikan,
                "Jam": jam,
                "Bagian": bagian

            }

            row.update(feature_data)

            all_data.append(row)

        except Exception as e:

            print(f"\nError : {file}")
            print(e)

# =========================================================
# DATAFRAME
# =========================================================

df = pd.DataFrame(all_data)

# =========================================================
# SIMPAN CSV
# =========================================================

df.to_csv(

    OUTPUT_CSV,

    index=False

)

# =========================================================
# MENAMPILKAN HASIL
# =========================================================

print("\n===================================")
print("HASIL EKSTRAKSI FITUR")
print("===================================\n")

print(df.head())

# =========================================================
# INFORMASI
# =========================================================

print("\n===================================")
print("CSV BERHASIL DISIMPAN")
print("===================================\n")

print(OUTPUT_CSV)