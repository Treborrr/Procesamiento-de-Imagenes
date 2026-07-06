"""Genera la tabla comparativa final entre HOG+SVM (clasico) y ResNet50
(profundo) para la seccion de Publicacion de Resultados del informe.

Genera dos tablas: evaluacion por frame (cada frame es una muestra
independiente) y evaluacion por video (mayoria de votos entre los frames
de cada clip, igual que hace gui_app.py en inferencia real). Esta segunda
es la mas representativa del uso real del sistema.

Uso:
    python compare_models.py
"""

import json

import pandas as pd

from config import RESULTS_DIR

MODELS = {
    "HOG + SVM (clasico)": "hog_svm",
    "ResNet50 (profundo)": "resnet50",
}

RENAME = {
    "accuracy": "Accuracy",
    "precision_macro": "Precision (macro)",
    "recall_macro": "Recall (macro)",
    "f1_macro": "F1 (macro)",
    "precision_weighted": "Precision (weighted)",
    "recall_weighted": "Recall (weighted)",
    "f1_weighted": "F1 (weighted)",
}


def build_table(suffix: str) -> pd.DataFrame:
    rows = []
    for label, model_name in MODELS.items():
        with open(RESULTS_DIR / f"{model_name}{suffix}_metrics.json", encoding="utf-8") as f:
            metrics = json.load(f)
        rows.append({"Modelo": label, **metrics})
    return pd.DataFrame(rows).set_index("Modelo").rename(columns=RENAME).round(4)


def show_and_save(df: pd.DataFrame, title: str, basename: str) -> None:
    print(f"\n=== {title} ===\n")
    print(df.to_string())

    out_csv = RESULTS_DIR / f"{basename}.csv"
    df.to_csv(out_csv)
    out_md = RESULTS_DIR / f"{basename}.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(df.to_markdown())
    print(f"\nGuardado en: {out_csv} / {out_md}")

    best = df["Accuracy"].idxmax()
    print(f"Mejor modelo por accuracy: {best} ({df.loc[best, 'Accuracy']:.1%})")


def main() -> None:
    frame_df = build_table(suffix="")
    show_and_save(frame_df, "Tabla comparativa por FRAME", "comparison_table")

    video_df = build_table(suffix="_video_level")
    show_and_save(video_df, "Tabla comparativa por VIDEO (mayoria de votos, uso real)", "comparison_table_video_level")


if __name__ == "__main__":
    main()
