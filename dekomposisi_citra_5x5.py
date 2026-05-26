# =========================================================
# VISUALISASI CITRA PIXELATED IKAN KEMBUNG
# DEKOMPOSISI CITRA RGB 5x5
# =========================================================

# =========================================================
# IMPORT LIBRARY
# =========================================================

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

# =========================================================
# INPUT CITRA
# =========================================================

IMAGE_PATH = r"C:\Rangga\SKRIPSI\BILA\Website\ikan_kembung.jpg"

# =========================================================
# FOLDER OUTPUT
# =========================================================

OUTPUT_FOLDER = r"C:\Rangga\SKRIPSI\BILA\Website\hasil_pengolahan"

ORIGINAL_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "citra_asli"
)

PIXEL_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "visualisasi_pixelated"
)

RED_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "channel_red"
)

GREEN_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "channel_green"
)

BLUE_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "channel_blue"
)

# =========================================================
# MEMBUAT FOLDER OTOMATIS
# =========================================================

os.makedirs(ORIGINAL_FOLDER, exist_ok=True)
os.makedirs(PIXEL_FOLDER, exist_ok=True)
os.makedirs(RED_FOLDER, exist_ok=True)
os.makedirs(GREEN_FOLDER, exist_ok=True)
os.makedirs(BLUE_FOLDER, exist_ok=True)

# =========================================================
# MEMBACA CITRA
# =========================================================

image = Image.open(
    IMAGE_PATH
).convert("RGB")

# =========================================================
# RESIZE CITRA ASLI
# =========================================================

IMG_SIZE = 224

image = image.resize(
    (IMG_SIZE, IMG_SIZE)
)

# =========================================================
# CONVERT NUMPY
# =========================================================

image_np = np.array(image)

# =========================================================
# SIMPAN CITRA ASLI
# =========================================================

original_save = os.path.join(
    ORIGINAL_FOLDER,
    "citra_asli.png"
)

cv2.imwrite(

    original_save,

    cv2.cvtColor(
        image_np,
        cv2.COLOR_RGB2BGR
    )

)

# =========================================================
# MEMBUAT EFEK PIXELATED
# =========================================================

# memperkecil citra
small_image = cv2.resize(

    image_np,

    (32, 32),

    interpolation=cv2.INTER_LINEAR

)

# memperbesar kembali dengan nearest neighbor
pixelated_image = cv2.resize(

    small_image,

    (500, 500),

    interpolation=cv2.INTER_NEAREST

)

# =========================================================
# SIMPAN VISUALISASI PIXELATED
# =========================================================

pixelated_save = os.path.join(
    PIXEL_FOLDER,
    "visualisasi_pixelated.png"
)

cv2.imwrite(

    pixelated_save,

    cv2.cvtColor(
        pixelated_image,
        cv2.COLOR_RGB2BGR
    )

)

# =========================================================
# MENGAMBIL SAMPEL 5x5 PIXEL
# =========================================================

sample_5x5 = image_np[0:5, 0:5]

# =========================================================
# MEMISAHKAN CHANNEL RGB
# =========================================================

red_channel = sample_5x5[:, :, 0]

green_channel = sample_5x5[:, :, 1]

blue_channel = sample_5x5[:, :, 2]

# =========================================================
# MEMBUAT DATAFRAME RGB
# =========================================================

red_df = pd.DataFrame(red_channel)

green_df = pd.DataFrame(green_channel)

blue_df = pd.DataFrame(blue_channel)

# =========================================================
# SIMPAN TABEL RGB
# =========================================================

red_df.to_csv(

    os.path.join(
        RED_FOLDER,
        "red_channel.csv"
    ),

    index=False

)

green_df.to_csv(

    os.path.join(
        GREEN_FOLDER,
        "green_channel.csv"
    ),

    index=False

)

blue_df.to_csv(

    os.path.join(
        BLUE_FOLDER,
        "blue_channel.csv"
    ),

    index=False

)

# =========================================================
# MENAMPILKAN CITRA ASLI
# =========================================================

plt.figure(figsize=(6,6))

plt.imshow(image_np)

plt.title("Citra Asli Ikan Kembung")

plt.axis("off")

plt.show()

# =========================================================
# MENAMPILKAN PIXELATED IMAGE
# =========================================================

plt.figure(figsize=(6,6))

plt.imshow(pixelated_image)

plt.title("Visualisasi Pixelated Image")

plt.axis("off")

plt.show()

# =========================================================
# MENAMPILKAN TABEL RGB
# =========================================================

print("\n===================================")
print("TABEL RED CHANNEL 5x5")
print("===================================\n")

print(red_df)

print("\n===================================")
print("TABEL GREEN CHANNEL 5x5")
print("===================================\n")

print(green_df)

print("\n===================================")
print("TABEL BLUE CHANNEL 5x5")
print("===================================\n")

print(blue_df)

# =========================================================
# INFORMASI PENYIMPANAN
# =========================================================

print("\n===================================")
print("HASIL BERHASIL DISIMPAN")
print("===================================\n")

print("Folder Output :")
print(OUTPUT_FOLDER)

print("\nIsi Folder :")
print("- citra_asli/")
print("- visualisasi_pixelated/")
print("- channel_red/")
print("- channel_green/")
print("- channel_blue/")