# =========================================================
# CEK PARAMETER MODEL SVM
# =========================================================

import os
import joblib

# =========================================================
# PATH MODEL
# =========================================================

MODEL_PATH = r"C:\Rangga\SKRIPSI\BILA\Website\model\svm_mobilenetv3.pkl"

# =========================================================
# LOAD MODEL
# =========================================================

print("\n===================================")
print("MEMUAT MODEL SVM")
print("===================================\n")

svm_model = joblib.load(MODEL_PATH)

print("Model berhasil dimuat")

# =========================================================
# PARAMETER SVM
# =========================================================

print("\n===================================")
print("PARAMETER MODEL SVM")
print("===================================\n")

# =========================================================
# KERNEL
# =========================================================

kernel = svm_model.kernel

print(f"Kernel              : {kernel}")

# =========================================================
# C (REGULARIZATION)
# =========================================================

c_value = svm_model.C

print(f"C (Regularization)  : {c_value}")

# =========================================================
# GAMMA
# =========================================================

gamma_value = svm_model.gamma

print(f"Gamma               : {gamma_value}")

# =========================================================
# JUMLAH FITUR
# =========================================================

jumlah_fitur = svm_model.support_vectors_.shape[1]

print(f"Jumlah Fitur        : {jumlah_fitur}")

# =========================================================
# JUMLAH KELAS
# =========================================================

jumlah_kelas = len(svm_model.classes_)

print(f"Jumlah Kelas        : {jumlah_kelas}")

# =========================================================
# NAMA KELAS
# =========================================================

print("\nKelas:")

for kelas in svm_model.classes_:

    print(f"- {kelas}")

# =========================================================
# RINGKASAN TABEL SKRIPSI
# =========================================================

print("\n===================================")
print("RINGKASAN UNTUK TABEL SKRIPSI")
print("===================================\n")

print(f"Kernel              : {kernel}")
print(f"C (Regularization)  : {c_value}")
print(f"Gamma               : {gamma_value}")
print(f"Jumlah Fitur        : {jumlah_fitur}")
print(f"Jumlah Kelas        : {jumlah_kelas}")

print("\n===================================")
print("SELESAI")
print("===================================\n")