import os
import base64
import cv2 
import joblib
import numpy as np

from flask import Flask, render_template, request
from PIL import Image
from io import BytesIO

from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model

# =====================================================
# FLASK SETUP
# =====================================================

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# =====================================================
# LOAD ARTIFACTS
# =====================================================

print("\n[INFO] Memuat model SVM...")
svm_model = joblib.load("model/svm_mobilenetv3.pkl")
print("[INFO] Model SVM berhasil dimuat.")

print("[INFO] Memuat scaler...")
scaler = joblib.load("model/scaler.pkl")
print("[INFO] Scaler berhasil dimuat.")

print("[INFO] Memuat MobileNetV3 Small...")
base_model = MobileNetV3Small(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
x = GlobalAveragePooling2D()(base_model.output)
feature_extractor = Model(inputs=base_model.input, outputs=x)
print("[INFO] MobileNetV3 Small berhasil dimuat.\n")

IMG_SIZE = 224
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}

# =====================================================
# FEATURE EXTRACTION
# =====================================================

def extract_color_features(image):
    mean_r = np.mean(image[:, :, 0])
    mean_g = np.mean(image[:, :, 1])
    mean_b = np.mean(image[:, :, 2])

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h_mean = np.mean(hsv[:, :, 0])
    s_mean = np.mean(hsv[:, :, 1])
    v_mean = np.mean(hsv[:, :, 2])

    brightness = np.mean(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
    redness = mean_r - ((mean_g + mean_b) / 2)

    return np.array([mean_r, mean_g, mean_b, h_mean, s_mean, v_mean, brightness, redness])


def extract_texture_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0)
    contrast = gray.std()
    return np.array([laplacian_var, edge_density, contrast])


def extract_cnn_feature(image):
    img = image.astype(np.float32) / 255.0
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    feature = feature_extractor.predict(img, verbose=0)
    return feature.flatten()


def extract_final_feature(image):
    cnn_feat     = extract_cnn_feature(image)
    color_feat   = extract_color_features(image)
    texture_feat = extract_texture_features(image)
    return np.concatenate([cnn_feat, color_feat, texture_feat])


def load_image_from_pil(pil_image):
    """Resize PIL Image to IMG_SIZE and return numpy array."""
    pil_image = pil_image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    return np.array(pil_image)

# =====================================================
# ROUTES
# =====================================================

@app.route('/', methods=['GET', 'POST'])
def index():
    result         = None
    confidence     = None
    uploaded_image = None
    error_message  = None

    if request.method == 'POST':
        try:
            source = request.form.get('source', 'upload')
            fish_path = os.path.join(app.config['UPLOAD_FOLDER'], "fish.jpg")

            if source == 'camera':
                # ---- Camera path: base64 data URI from canvas ----
                camera_data = request.form.get('camera_data', '')
                if not camera_data:
                    raise ValueError("Tidak ada data foto dari kamera.")

                # Strip the data URI header (e.g. "data:image/jpeg;base64,...")
                if ',' in camera_data:
                    camera_data = camera_data.split(',', 1)[1]

                img_bytes = base64.b64decode(camera_data)
                pil_image = Image.open(BytesIO(img_bytes))
                pil_image.convert("RGB").save(fish_path, "JPEG", quality=90)
                uploaded_image = "uploads/fish.jpg"
                image = load_image_from_pil(pil_image)

            else:
                # ---- Upload path ----
                fish = request.files.get('fish')

                if not fish or fish.filename == '':
                    raise ValueError("Belum ada gambar yang dipilih.")

                ext = os.path.splitext(fish.filename)[1].lower()
                if ext not in ALLOWED_EXTENSIONS:
                    raise ValueError(
                        f"Format '{ext}' tidak didukung. Gunakan JPG, PNG, atau WEBP."
                    )

                fish.save(fish_path)
                uploaded_image = "uploads/fish.jpg"

                pil_image = Image.open(fish_path)
                image = load_image_from_pil(pil_image)

            # ---- Feature extraction & prediction (shared) ----
            feature = extract_final_feature(image)
            feature_scaled = scaler.transform([feature])

            prediction  = svm_model.predict(feature_scaled)[0]
            probability = svm_model.predict_proba(feature_scaled)[0]
            confidence  = round(float(np.max(probability)) * 100, 2)

            if confidence < 65:
                result = "BUKAN_IKAN"
            elif prediction == 1:
                result = "SEGAR"
            else:
                result = "TIDAK_SEGAR"

        except ValueError as e:
            result        = "ERROR"
            error_message = str(e)
        except Exception as e:
            result        = "ERROR"
            error_message = f"Terjadi kesalahan saat memproses gambar: {str(e)}"

    return render_template(
        'index.html',
        result=result,
        confidence=confidence,
        uploaded_image=uploaded_image,
        error_message=error_message
    )


if __name__ == '__main__':
    app.run(debug=True)