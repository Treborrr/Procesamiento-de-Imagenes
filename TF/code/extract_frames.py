"""Sub-muestrea y redimensiona los frames ya extraidos en data/raw/<clase>/<video_id>/*.jpg
hacia data/frames/<clase>/<video_id>/frame_XXXX.jpg (224x224 RGB, hasta
MAX_FRAMES_PER_VIDEO frames por clip, distribuidos uniformemente).

El mirror de Kaggle usado ya trae, por cada clip, solo un puñado de frames
representativos (10-60 aprox., promedio ~15) en vez del video completo a fps
nativo, y no publica el fps de extraccion usado. Por eso NO se asume un fps
nativo para hacer un subsampling por stride fijo (eso descartaria casi todos
los frames de clips ya cortos): en vez de eso, se toman TODOS los frames si
el video tiene <= MAX_FRAMES_PER_VIDEO, o una muestra uniforme en caso
contrario (ver DECISIONES_TECNICAS.md, seccion 6/7).

Los frames quedan organizados por video (no solo por clase) para permitir
un split train/test a nivel de video sin fuga de datos.

Tambien genera data/splits/frames_metadata.csv con columnas:
    frame_path, class, video_id

Uso:
    python extract_frames.py
"""

import csv

import cv2
import numpy as np
from tqdm import tqdm

from config import CLASSES, FRAME_SIZE, FRAMES_DIR, MAX_FRAMES_PER_VIDEO, RAW_DIR, SPLITS_DIR

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def subsample_video_frames(video_dir, out_dir) -> int:
    frame_files = sorted(
        p for p in video_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not frame_files:
        return 0

    if len(frame_files) <= MAX_FRAMES_PER_VIDEO:
        sampled = frame_files
    else:
        indices = np.linspace(0, len(frame_files) - 1, MAX_FRAMES_PER_VIDEO).round().astype(int)
        sampled = [frame_files[i] for i in indices]

    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for i, frame_path in enumerate(sampled):
        img = cv2.imread(str(frame_path))
        if img is None:
            continue
        img = cv2.resize(img, FRAME_SIZE, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out_dir / f"frame_{i:04d}.jpg"), img)
        saved += 1
    return saved


def main() -> None:
    rows = []

    for class_name in CLASSES:
        class_raw_dir = RAW_DIR / class_name
        video_dirs = [p for p in class_raw_dir.iterdir() if p.is_dir()] if class_raw_dir.exists() else []
        if not video_dirs:
            print(f"ADVERTENCIA: no hay videos en {class_raw_dir}")
            continue

        for video_dir in tqdm(video_dirs, desc=f"Sub-muestreando frames [{class_name}]"):
            video_id = video_dir.name
            out_dir = FRAMES_DIR / class_name / video_id
            n_saved = subsample_video_frames(video_dir, out_dir)
            for i in range(n_saved):
                rows.append({
                    "frame_path": str((out_dir / f"frame_{i:04d}.jpg").relative_to(FRAMES_DIR)),
                    "class": class_name,
                    "video_id": video_id,
                })

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    metadata_path = SPLITS_DIR / "frames_metadata.csv"
    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_path", "class", "video_id"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nTotal frames extraidos: {len(rows)}")
    print(f"Metadata guardada en: {metadata_path}")


if __name__ == "__main__":
    main()
