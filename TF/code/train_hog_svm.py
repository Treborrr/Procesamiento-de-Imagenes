"""Entrena y evalua el modelo clasico HOG + SVM.

Pipeline (seccion 3.2/3.4 del informe): fotograma -> escala de grises ->
redimension 64x128 -> descriptor HOG (celdas 8x8, bloques 2x2, 9 orientaciones,
~3780 valores) -> SVM kernel RBF, estrategia One-vs-Rest.

Uso:
    python train_hog_svm.py
"""

import cv2
import joblib
import numpy as np
import pandas as pd
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

from config import (
    FRAMES_DIR,
    HOG_CELLS_PER_BLOCK,
    HOG_IMAGE_SIZE,
    HOG_ORIENTATIONS,
    HOG_PIXELS_PER_CELL,
    MODELS_DIR,
    SPLITS_DIR,
    SVM_C,
    SVM_GAMMA,
    SVM_KERNEL,
)
from eval_utils import report, report_video_level

MODEL_NAME = "hog_svm"


def extract_hog_features(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    height, width = HOG_IMAGE_SIZE
    gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)
    return hog(
        gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm="L2-Hys",
    )


def build_feature_matrix(df: pd.DataFrame, desc: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features, labels, video_ids = [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        img_path = FRAMES_DIR / row["frame_path"]
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        features.append(extract_hog_features(img))
        labels.append(row["class"])
        video_ids.append(row["video_id"])
    return np.array(features), np.array(labels), np.array(video_ids)


def main() -> None:
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    test_df = pd.read_csv(SPLITS_DIR / "test.csv")

    X_train, y_train, _ = build_feature_matrix(train_df, "Extrayendo HOG [train]")
    X_test, y_test, video_ids_test = build_feature_matrix(test_df, "Extrayendo HOG [test]")

    print(f"\nDimension del vector HOG: {X_train.shape[1]}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # class_weight="balanced" compensa que 'run' tiene ~2x mas videos que
    # 'pick' en el dataset (232 vs 106): sin esto, el SVM tiende a sesgarse
    # hacia la clase mayoritaria (ver DECISIONES_TECNICAS.md, seccion 12).
    print("Entrenando SVM (kernel RBF, One-vs-Rest, class_weight=balanced)...")
    svm = SVC(
        kernel=SVM_KERNEL,
        C=SVM_C,
        gamma=SVM_GAMMA,
        decision_function_shape="ovr",
        probability=True,
        class_weight="balanced",
        random_state=42,
    )
    svm.fit(X_train_scaled, y_train)

    y_pred = svm.predict(X_test_scaled)

    print("\n=== Resultados HOG + SVM (por frame) ===")
    report(y_test, y_pred, MODEL_NAME)
    report_video_level(video_ids_test, y_test, y_pred, MODEL_NAME)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": svm, "scaler": scaler}, MODELS_DIR / "hog_svm.pkl")
    print(f"\nModelo guardado en: {MODELS_DIR / 'hog_svm.pkl'}")


if __name__ == "__main__":
    main()
