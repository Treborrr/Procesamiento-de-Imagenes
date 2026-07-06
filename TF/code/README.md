# Código fuente — Trabajo Final PI

Pipeline completo, en orden de ejecución:

```bash
pip install -r requirements.txt
# PyTorch (CPU) se instala aparte:
pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cpu

python download_dataset.py   # descarga HMDB51 de Kaggle y filtra pick/run/stand -> data/raw/
python extract_frames.py     # sub-muestrea y redimensiona frames -> data/frames/
python make_splits.py        # split train/test por video -> data/splits/{train,test}.csv

python train_hog_svm.py      # entrena y evalua el modelo clasico -> models/hog_svm.pkl
python train_resnet50.py     # entrena y evalua el modelo profundo -> models/resnet50.pt
python compare_models.py     # tabla comparativa final -> results/comparison_table.{csv,md}

python gui_app.py            # GUI (Gradio) en http://127.0.0.1:7860, requiere los .pkl/.pt entrenados
```

## Requisitos

- Cuenta de Kaggle con API configurada (`kaggle login`, o `~/.kaggle/kaggle.json`).
- Sin GPU: todo el pipeline corre en CPU (ResNet50 usa embeddings cacheados del backbone
  congelado para que el entrenamiento sea rapido, ver `DECISIONES_TECNICAS.md` seccion 8).

## Estructura de archivos

| Archivo | Rol |
|---|---|
| `config.py` | Rutas, clases, hiperparametros centralizados |
| `download_dataset.py` | Descarga HMDB51 y filtra clases del caso de uso |
| `extract_frames.py` | Muestreo/resize de frames por video |
| `make_splits.py` | Split train/test estratificado por video (sin fuga de datos) |
| `train_hog_svm.py` | Modelo clasico: HOG + SVM |
| `train_resnet50.py` | Modelo profundo: ResNet50 + transfer learning |
| `eval_utils.py` | Metricas, matriz de confusion y reporte por clase compartidos |
| `compare_models.py` | Tabla comparativa final entre ambos modelos |
| `gui_app.py` | Interfaz Gradio para el usuario final |

## Salidas generadas

- `models/hog_svm.pkl`, `models/resnet50.pt` — modelos entrenados
- `results/*_metrics.json` — metricas agregadas (accuracy, precision/recall/f1 macro y weighted)
- `results/*_per_class_metrics.csv` — metricas por clase (pick/run/stand)
- `results/*_confusion_matrix.png` — matrices de confusion
- `results/*_predictions.csv` — predicciones crudas (y_true/y_pred) del set de test
- `results/comparison_table.{csv,md}` — tabla comparativa HOG+SVM vs. ResNet50

Ver `../DECISIONES_TECNICAS.md` para el historial de decisiones tecnicas (antes/ahora/por que)
tomadas durante el desarrollo, útil para redactar el informe final.
