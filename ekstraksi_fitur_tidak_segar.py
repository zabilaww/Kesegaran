# =========================================================
# EKSTRAKSI FITUR IKAN KEMBUNG TIDAK SEGAR
# MobileNetV3 Small
# =========================================================

# =========================================================
# IMPORT LIBRARY
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

OUTPUT_CSV = r"C:\Rangga\SKRIPSI\BILA\Website\hasil_ekstraksi_tidak_segar.csv"

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
# DATASET FOLDER
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
    # FLOAT
    # =============================================

    image = image.astype(
        np.float32
    )

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
print("MEMBACA DATASET TIDAK SEGAR")
print("===================================\n")

for dataset_name in dataset_folders:

    # =============================================
    # FOLDER TIDAK SEGAR
    # =============================================

    tidak_segar_folder = os.path.join(

        DATASET_PATH,

        dataset_name,

        "tidak_segar"

    )

    if not os.path.exists(tidak_segar_folder):
        continue

    # =============================================
    # AMBIL FILE GAMBAR
    # =============================================

    files = [

        f for f in os.listdir(tidak_segar_folder)

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
            tidak_segar_folder,
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

                # jam
                elif p.isdigit():

                    jam = p

                # bagian
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
            # SIMPAN FITUR
            # =====================================

            row = {

                "Dataset": dataset_name,
                "Ikan": ikan,
                "Jam": jam,
                "Bagian": bagian

            }

            # =====================================
            # F1 - F50
            # =====================================

            for i in range(50):

                row[f"F{i+1}"] = round(

                    float(feature[i]),

                    6

                )

            # =====================================
            # FITUR TERAKHIR
            # =====================================

            row["F576"] = round(

                float(feature[-1]),

                6

            )

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