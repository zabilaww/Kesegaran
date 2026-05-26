# =========================================================
# FILE : prepare_dataset_jenis.py
# =========================================================

import os
import cv2
import numpy as np

# =========================================================
# PATH
# =========================================================

DATASET_AWAL = r"C:\Rangga\SKRIPSI\BILA\Website\dataset\dataset_awal"

DATASET_JENIS = r"C:\Rangga\SKRIPSI\BILA\Website\dataset\dataset_jenis"

DATASET_BUKAN_KEMBUNG = DATASET_BUKAN_KEMBUNG = r"C:\Rangga\SKRIPSI\BILA\Website\dataset\bukan_kembung_raw"

# =========================================================
# OUTPUT FOLDER
# =========================================================

KEMBUNG_SEGAR = os.path.join(
    DATASET_JENIS,
    "ikan_kembung",
    "segar"
)

KEMBUNG_TIDAK_SEGAR = os.path.join(
    DATASET_JENIS,
    "ikan_kembung",
    "tidak_segar"
)

BUKAN_KEMBUNG = os.path.join(
    DATASET_JENIS,
    "bukan_ikan_kembung"
)

# =========================================================
# MEMBUAT FOLDER
# =========================================================

os.makedirs(
    KEMBUNG_SEGAR,
    exist_ok=True
)

os.makedirs(
    KEMBUNG_TIDAK_SEGAR,
    exist_ok=True
)

os.makedirs(
    BUKAN_KEMBUNG,
    exist_ok=True
)

# =========================================================
# EXTENSION VALID
# =========================================================

VALID_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".JPG",
    ".JPEG",
    ".PNG"
)

# =========================================================
# IMAGE SIZE
# =========================================================

IMG_SIZE = 224

# =========================================================
# AI SMART PREPROCESS
# =========================================================

def preprocess_ai(image):

    # =============================================
    # RESIZE
    # =============================================

    image = cv2.resize(
        image,
        (IMG_SIZE, IMG_SIZE)
    )

    # =============================================
    # DENOISE RINGAN
    # =============================================

    image = cv2.fastNlMeansDenoisingColored(
        image,
        None,
        3,
        3,
        7,
        21
    )

    # =============================================
    # CLAHE
    # MEMPERJELAS WARNA DAN TEKSTUR
    # =============================================

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8,8)
    )

    cl = clahe.apply(l)

    limg = cv2.merge((cl,a,b))

    image = cv2.cvtColor(
        limg,
        cv2.COLOR_LAB2BGR
    )

    return image

# =========================================================
# MEMPROSES IKAN KEMBUNG
# =========================================================

print("\n===================================")
print("MEMPROSES IKAN KEMBUNG")
print("===================================\n")

total_kembung = 0

# =========================================================
# FOLDER:
# Segar
# Tidak Segar
# =========================================================

for label in ["Segar", "tidak Segar"]:

    source_folder = os.path.join(
        DATASET_AWAL,
        label
    )

    # =============================================
    # VALIDASI FOLDER
    # =============================================

    if not os.path.exists(source_folder):

        print(f"\nFOLDER TIDAK DITEMUKAN:")
        print(source_folder)

        continue

    # =============================================
    # LABEL OUTPUT
    # =============================================

    if label == "Segar":

        destination_folder = KEMBUNG_SEGAR

    else:

        destination_folder = KEMBUNG_TIDAK_SEGAR

    # =============================================
    # MEMBACA FILE
    # =============================================

    for file in os.listdir(source_folder):

        if not file.lower().endswith(
            VALID_EXTENSIONS
        ):
            continue

        source_path = os.path.join(
            source_folder,
            file
        )

        # =========================================
        # MEMBACA GAMBAR
        # =========================================

        image = cv2.imread(source_path)

        if image is None:

            print(f"[SKIP CORRUPT] {file}")

            continue

        # =========================================
        # AI PREPROCESS
        # =========================================

        image = preprocess_ai(image)

        # =========================================
        # SAVE
        # =========================================

        save_path = os.path.join(
            destination_folder,
            file
        )

        cv2.imwrite(
            save_path,
            image
        )

        total_kembung += 1

        print(f"[KEMBUNG] {file}")

# =========================================================
# MEMPROSES BUKAN IKAN KEMBUNG
# =========================================================

print("\n===================================")
print("MEMPROSES BUKAN IKAN KEMBUNG")
print("===================================\n")

total_bukan = 0

if os.path.exists(DATASET_BUKAN_KEMBUNG):

    for file in os.listdir(DATASET_BUKAN_KEMBUNG):

        if not file.lower().endswith(
            VALID_EXTENSIONS
        ):
            continue

        source_path = os.path.join(
            DATASET_BUKAN_KEMBUNG,
            file
        )

        image = cv2.imread(source_path)

        if image is None:

            print(f"[SKIP CORRUPT] {file}")

            continue

        # =========================================
        # AI PREPROCESS
        # =========================================

        image = preprocess_ai(image)

        # =========================================
        # SAVE
        # =========================================

        save_path = os.path.join(
            BUKAN_KEMBUNG,
            file
        )

        cv2.imwrite(
            save_path,
            image
        )

        total_bukan += 1

        print(f"[BUKAN KEMBUNG] {file}")

else:

    print("\nFOLDER dataset_bukan_kembung BELUM ADA")

# =========================================================
# HASIL
# =========================================================

print("\n===================================")
print("DATASET JENIS BERHASIL DIBUAT")
print("===================================")

print(f"\nIKAN KEMBUNG      : {total_kembung}")

print(f"BUKAN KEMBUNG     : {total_bukan}")

print(f"\nLokasi dataset :")
print(DATASET_JENIS)

print("\n===================================")