"""Utilidades de evaluacion compartidas por ambos modelos (HOG+SVM y ResNet50),
para que produzcan metricas en el mismo formato y se puedan comparar directamente
en la seccion de Publicacion de Resultados del informe.
"""

import json

import matplotlib
matplotlib.use("Agg")  # solo se guardan graficos a disco, sin backend interactivo (Tk)
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from config import CLASSES, RESULTS_DIR


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def save_per_class_report(y_true, y_pred, model_name: str) -> pd.DataFrame:
    """Desglose de precision/recall/f1/support por clase (pick/run/stand),
    complementario a las metricas macro/weighted agregadas."""
    report_dict = classification_report(
        y_true, y_pred, labels=CLASSES, output_dict=True, zero_division=0,
    )
    df = pd.DataFrame(report_dict).T.loc[CLASSES]
    df = df.rename(columns={
        "precision": "Precision", "recall": "Recall", "f1-score": "F1", "support": "Support",
    })

    out_csv = RESULTS_DIR / f"{model_name}_per_class_metrics.csv"
    df.round(4).to_csv(out_csv)
    print(f"Metricas por clase guardadas en: {out_csv}")
    print(df.round(4).to_string())
    return df


def save_confusion_matrix(y_true, y_pred, model_name: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel("Prediccion")
    plt.ylabel("Real")
    plt.title(f"Matriz de confusion - {model_name}")
    plt.tight_layout()
    out_path = RESULTS_DIR / f"{model_name}_confusion_matrix.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Matriz de confusion guardada en: {out_path}")


def save_metrics(metrics: dict, model_name: str) -> None:
    out_path = RESULTS_DIR / f"{model_name}_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Metricas guardadas en: {out_path}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


def save_predictions(y_true, y_pred, model_name: str) -> None:
    """Guarda las predicciones crudas para poder recalcular metricas despues
    sin tener que re-entrenar el modelo."""
    out_path = RESULTS_DIR / f"{model_name}_predictions.csv"
    pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(out_path, index=False)
    print(f"Predicciones guardadas en: {out_path}")


def report(y_true, y_pred, model_name: str) -> dict:
    metrics = compute_metrics(y_true, y_pred)
    save_metrics(metrics, model_name)
    save_per_class_report(y_true, y_pred, model_name)
    save_confusion_matrix(y_true, y_pred, model_name)
    save_predictions(y_true, y_pred, model_name)
    return metrics


def aggregate_by_video(video_ids, y_true, y_pred) -> tuple[list, list]:
    """Agrega predicciones de frame a nivel de video: todas las filas de un
    mismo video_id comparten la clase real, y la prediccion del video es la
    clase mas votada entre sus frames (igual que hace gui_app.py en
    inferencia real). Evaluar asi -en vez de por frame suelto- refleja mejor
    el uso real del sistema (se sube un clip, no un frame aislado)."""
    df = pd.DataFrame({"video_id": video_ids, "y_true": y_true, "y_pred": y_pred})
    agg = df.groupby("video_id").agg(
        y_true=("y_true", lambda s: s.mode().iloc[0]),
        y_pred=("y_pred", lambda s: s.mode().iloc[0]),
    )
    return agg["y_true"].tolist(), agg["y_pred"].tolist()


def report_video_level(video_ids, y_true, y_pred, model_name: str) -> dict:
    video_true, video_pred = aggregate_by_video(video_ids, y_true, y_pred)
    print(f"\n--- Evaluacion a nivel de video ({len(video_true)} videos, mayoria de votos) ---")
    return report(video_true, video_pred, f"{model_name}_video_level")
