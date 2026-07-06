"""Configuracion central del proyecto: rutas, clases e hiperparametros.

Caso de uso: clasificacion de productividad de empleados en almacen
(Pick Up / Running / Standing) via computer vision.

Nota sobre el dataset: el Hito 1 referenciaba UCF101, pero UCF101 no
contiene clases de "persona parada/inactiva" (sus 101 clases son
deportes, instrumentos e interacciones). El dataset correcto que si
tiene las 3 clases del caso de uso (pick, run, stand) es HMDB51.
Este cambio debe reflejarse en la seccion de dataset del informe final.
"""

from pathlib import Path

# --- Rutas ---
CODE_DIR = Path(__file__).resolve().parent
TF_DIR = CODE_DIR.parent
DATA_DIR = TF_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
FRAMES_DIR = DATA_DIR / "frames"
SPLITS_DIR = DATA_DIR / "splits"
MODELS_DIR = TF_DIR / "models"
RESULTS_DIR = TF_DIR / "results"

for d in (RAW_DIR, FRAMES_DIR, SPLITS_DIR, MODELS_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# --- Dataset (Kaggle) ---
KAGGLE_DATASET_SLUG = "easonlll/hmdb51"  # mirror alternativo: "jizeyong/hmdb51"

# Clases HMDB51 reales -> etiqueta de negocio del caso de uso
CLASSES = ["pick", "run", "stand"]
CLASS_TO_BUSINESS_LABEL = {
    "pick": "Activo (Pick Up)",
    "run": "Activo (Running)",
    "stand": "Inactivo/Pausado (Standing)",
}

# --- Extraccion de frames ---
# El mirror de Kaggle usado ya entrega pocos frames representativos por clip
# (10-60, promedio ~15) y no publica el fps de extraccion, por lo que ya no
# se aplica un muestreo por fps fijo (ver DECISIONES_TECNICAS.md, seccion 7):
# se toman todos los frames del clip, o una muestra uniforme si supera
# MAX_FRAMES_PER_VIDEO. FRAME_RATE_FPS se conserva solo como referencia del
# objetivo nominal del Hito 1 para la redaccion del informe.
FRAME_RATE_FPS = 5
FRAME_SIZE = (224, 224)  # ancho, alto - RGB, compatible con ambos modelos
MAX_FRAMES_PER_VIDEO = 30  # limite practico para entrenar en CPU

# --- Split ---
TEST_SIZE = 0.2
RANDOM_SEED = 42

# --- HOG + SVM (modelo clasico) ---
HOG_IMAGE_SIZE = (128, 64)  # alto, ancho (skimage usa alto x ancho)
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
SVM_KERNEL = "rbf"
SVM_C = 10.0
SVM_GAMMA = "scale"

# --- ResNet50 (modelo profundo) ---
RESNET_IMAGE_SIZE = (224, 224)
RESNET_BATCH_SIZE = 32
RESNET_EPOCHS = 12
RESNET_LR = 0.001
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
