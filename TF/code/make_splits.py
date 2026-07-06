"""Genera el split train/test a partir de data/splits/frames_metadata.csv.

El split se hace a nivel de VIDEO (no de frame): todos los frames de un mismo
clip quedan en el mismo conjunto (train o test). Si se dividiera por frame,
frames casi identicos del mismo video apareceria tanto en train como en test,
lo que produce fuga de datos (data leakage) y una accuracy de evaluacion
artificialmente inflada.

La proporcion de clases se mantiene igual en train y test (split estratificado).

Uso:
    python make_splits.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from config import RANDOM_SEED, SPLITS_DIR, TEST_SIZE


def main() -> None:
    metadata_path = SPLITS_DIR / "frames_metadata.csv"
    df = pd.read_csv(metadata_path)

    videos = df[["video_id", "class"]].drop_duplicates().reset_index(drop=True)

    train_videos, test_videos = train_test_split(
        videos,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=videos["class"],
    )

    train_ids = set(train_videos["video_id"])
    test_ids = set(test_videos["video_id"])

    train_df = df[df["video_id"].isin(train_ids)].reset_index(drop=True)
    test_df = df[df["video_id"].isin(test_ids)].reset_index(drop=True)

    train_df.to_csv(SPLITS_DIR / "train.csv", index=False)
    test_df.to_csv(SPLITS_DIR / "test.csv", index=False)

    print("Split por video (sin fuga de datos entre train/test):")
    print(f"  Videos train: {len(train_ids)} | frames train: {len(train_df)}")
    print(f"  Videos test:  {len(test_ids)} | frames test:  {len(test_df)}")
    print("\nDistribucion de clases:")
    print("  Train:\n", train_df["class"].value_counts())
    print("  Test:\n", test_df["class"].value_counts())


if __name__ == "__main__":
    main()
