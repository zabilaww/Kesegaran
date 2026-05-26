import matplotlib.pyplot as plt

# ==========================================
# DATASET
# ==========================================

kategori = [

    'Data Training',
    'Data Validasi',
    'Data Testing'

]

jumlah = [

    630,
    120,
    150

]

# ==========================================
# MEMBUAT GRAFIK
# ==========================================

plt.figure(

    figsize=(8,5)

)

bars = plt.bar(

    kategori,
    jumlah

)

# ==========================================
# MENAMPILKAN NILAI DI ATAS BATANG
# ==========================================

for bar in bars:

    yval = bar.get_height()

    plt.text(

        bar.get_x() + bar.get_width()/2,
        yval + 5,
        int(yval),

        ha='center'

    )

# ==========================================
# JUDUL DAN LABEL
# ==========================================

plt.title(

    'Distribusi Dataset Penelitian'

)

plt.xlabel(

    'Kategori Dataset'

)

plt.ylabel(

    'Jumlah Citra'

)

plt.grid(

    axis='y',
    linestyle='--',
    alpha=0.5

)

# ==========================================
# SIMPAN GAMBAR
# ==========================================

plt.savefig(

    'grafik_distribusi_dataset.png',
    dpi=300,
    bbox_inches='tight'

)

# ==========================================
# TAMPILKAN GRAFIK
# ==========================================

plt.show()