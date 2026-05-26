import os
import cv2
import numpy as np

# =========================================================
# PREPROCESSING DAN AUGMENTASI DATASET IKAN KEMBUNG
# FIX FINAL VERSION
# =========================================================

# =========================================================
# PATH DATASET
# =========================================================

DATASET_AWAL = r"C:\Bila\bila (iini folder yg trainingnya bagus) - Salin\dataset\dataset_awal"

DATASET_SPLIT = r"C:\Bila\bila (iini folder yg trainingnya bagus) - Salin\dataset\dataset_split"

# =========================================================
# MEMBUAT STRUKTUR FOLDER
# =========================================================

folders = [

    # TRAIN
    r"train\segar",
    r"train\tidak_segar",

    # VALID
    r"valid\segar",
    r"valid\tidak_segar",

    # TEST
    r"test\segar",
    r"test\tidak_segar"

]

for folder in folders:

    os.makedirs(
        os.path.join(DATASET_SPLIT, folder),
        exist_ok=True
    )

# =========================================================
# PARAMETER
# =========================================================

TARGET_WIDTH = 224
TARGET_HEIGHT = 224

# =========================================================
# JAM SEGAR
# =========================================================

JAM_SEGAR = [
    "07",
    "09",
    "11",
    "13",
    "15"
]

# =========================================================
# JAM TIDAK SEGAR
# =========================================================

JAM_TIDAK_SEGAR = [
    "17",
    "19",
    "21",
    "23",
    "01"
]

# =========================================================
# RESIZE
#
# x_asal = (x_baru × W_asal) / W_baru
# y_asal = (y_baru × H_asal) / H_baru
# =========================================================

def resize_image(image):

    return cv2.resize(
        image,
        (TARGET_WIDTH, TARGET_HEIGHT),
        interpolation=cv2.INTER_LINEAR
    )

# =========================================================
# NORMALISASI
#
# I_norm = I / 255
# =========================================================

def normalize_image(image):

    return image.astype(np.float32) / 255.0

# =========================================================
# ROTASI
#
# [x']   [ cosθ  -sinθ ] [x]
# [y'] = [ sinθ   cosθ ] [y]
# =========================================================

def rotate_image(image, angle):

    h, w = image.shape[:2]

    center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    return cv2.warpAffine(
        image,
        matrix,
        (w, h)
    )

# =========================================================
# FLIP HORIZONTAL
#
# x' = W - 1 - x
# y' = y
# =========================================================

def horizontal_flip(image):

    return cv2.flip(image, 1)

# =========================================================
# BLUR
# =========================================================

def blur_image(image):

    return cv2.GaussianBlur(
        image,
        (5, 5),
        0
    )

# =========================================================
# BRIGHTNESS
#
# I' = I + β
# =========================================================

def brightness_adjustment(image, beta=30):

    return cv2.convertScaleAbs(
        image,
        alpha=1.0,
        beta=beta
    )

# =========================================================
# MENENTUKAN SPLIT
# =========================================================

def determine_split(fish_number):

    # TRAIN
    if 1 <= fish_number <= 21:
        return "train"

    # VALID
    elif 22 <= fish_number <= 25:
        return "valid"

    # TEST
    elif 26 <= fish_number <= 30:
        return "test"

    else:
        return None

# =========================================================
# MENENTUKAN LABEL
# =========================================================

def determine_label(hour):

    if hour in JAM_SEGAR:
        return "segar"

    elif hour in JAM_TIDAK_SEGAR:
        return "tidak_segar"

    else:
        return None

# =========================================================
# COUNTER
# =========================================================

total_data = 0
skip_data = 0

# =========================================================
# MEMBACA DATASET
# =========================================================

for root, dirs, files in os.walk(DATASET_AWAL):

    for file in files:

        # =================================================
        # FORMAT FILE
        # =================================================

        if not file.lower().endswith(
            ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        ):
            continue

        image_path = os.path.join(root, file)

        # =================================================
        # MEMBACA GAMBAR
        # =================================================

        try:

            image = cv2.imread(image_path)

            if image is None:

                print(f"[SKIP CORRUPT] {file}")

                skip_data += 1

                continue

        except Exception as e:

            print(f"[ERROR] {file}")
            print(e)

            skip_data += 1

            continue

        # =================================================
        # NAMA FILE
        # =================================================

        filename = os.path.splitext(file)[0]

        # =================================================
        # FORMAT:
        # ikan01_mata_07
        # =================================================

        try:

            split_name = filename.split("_")

            fish_part = split_name[0]

            hour = split_name[-1]

            fish_number = int(
                fish_part.replace("ikan", "")
            )

        except:

            print(f"[SKIP FORMAT] {file}")

            skip_data += 1

            continue

        # =================================================
        # SPLIT
        # =================================================

        split = determine_split(fish_number)

        if split is None:

            skip_data += 1

            continue

        # =================================================
        # LABEL
        # =================================================

        label = determine_label(hour)

        if label is None:

            skip_data += 1

            continue

        # =================================================
        # FOLDER OUTPUT
        # =================================================

        save_folder = os.path.join(
            DATASET_SPLIT,
            split,
            label
        )

        os.makedirs(
            save_folder,
            exist_ok=True
        )

        # =================================================
        # PREPROCESSING
        # =================================================

        resized = resize_image(image)

        normalized = normalize_image(resized)

        # =================================================
        # KEMBALIKAN KE UINT8
        # =================================================

        processed_image = (
            normalized * 255
        ).astype(np.uint8)

        # =================================================
        # TRAIN → AUGMENTASI
        # =================================================

        if split == "train":

            augmentations = {

                "original":
                    processed_image,

                "flip_horizontal":
                    horizontal_flip(
                        processed_image
                    ),

                "rotasi_plus20":
                    rotate_image(
                        processed_image,
                        20
                    ),

                "rotasi_minus20":
                    rotate_image(
                        processed_image,
                        -20
                    ),

                "brightness":
                    brightness_adjustment(
                        processed_image,
                        30
                    ),

                "blur":
                    blur_image(
                        processed_image
                    )

            }

            # =============================================
            # SAVE AUGMENTATION
            # =============================================

            for aug_name, aug_image in augmentations.items():

                save_path = os.path.join(

                    save_folder,

                    f"{filename}_{aug_name}.jpg"

                )

                cv2.imwrite(
                    save_path,
                    aug_image
                )

                total_data += 1

        # =================================================
        # VALID & TEST → TANPA AUGMENTASI
        # =================================================

        else:

            save_path = os.path.join(

                save_folder,

                f"{filename}.jpg"

            )

            cv2.imwrite(
                save_path,
                processed_image
            )

            total_data += 1

        print(
            f"[BERHASIL] {split} | {label} | {file}"
        )

# =========================================================
# HASIL AKHIR
# =========================================================

print("\n======================================")
print("PREPROCESSING DAN AUGMENTASI SELESAI")
print("======================================")
print(f"Dataset awal      : {DATASET_AWAL}")
print(f"Dataset split     : {DATASET_SPLIT}")
print(f"Total berhasil    : {total_data}")
print(f"Total skip/error  : {skip_data}")
print("======================================")