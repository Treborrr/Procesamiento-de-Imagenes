"""Descarga el dataset HMDB51 desde Kaggle y filtra solo las clases del
caso de uso (pick, run, stand) hacia data/raw/<clase>/<video_id>/*.jpg

El mirror de Kaggle usado (easonlll/hmdb51) ya trae los videos con sus
frames pre-extraidos en JPG (uno por carpeta de video, a la resolucion y
fps nativos del clip original), en vez de archivos .avi. Por eso este
script copia carpetas de frames completas, y el resampleo a 5fps /
224x224 se hace despues en extract_frames.py.

Requiere credenciales de Kaggle en ~/.kaggle/kaggle.json (token clasico)
o ~/.kaggle/credentials.json (sesion de `kaggle login`).

Uso:
    python download_dataset.py
"""

import shutil
import zipfile
from pathlib import Path

from config import CLASSES, DATA_DIR, KAGGLE_DATASET_SLUG, RAW_DIR

DOWNLOAD_DIR = DATA_DIR / "_kaggle_download"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def check_credentials() -> None:
    kaggle_dir = Path.home() / ".kaggle"
    # El paquete `kaggle` acepta el token clasico (kaggle.json con
    # username/key) o la sesion OAuth generada por `kaggle login`
    # (credentials.json con access_token/refresh_token).
    if not (kaggle_dir / "kaggle.json").exists() and not (kaggle_dir / "credentials.json").exists():
        raise SystemExit(
            "No se encontraron credenciales de Kaggle.\n"
            "1. Entra a kaggle.com -> tu perfil -> Account -> Create New API Token\n"
            "2. Se descarga un archivo kaggle.json\n"
            f"3. Coloca ese archivo en: {kaggle_dir / 'kaggle.json'}\n"
            "4. Vuelve a correr este script."
        )


def download_and_extract() -> Path:
    from kaggle.api.kaggle_api_extended import KaggleApi

    # Idempotente: si ya se descargo/descomprimio antes, no repetir 3+ GB de descarga.
    existing_dirs = [d for d in DOWNLOAD_DIR.glob("*") if d.is_dir()]
    if existing_dirs:
        print(f"Ya existe contenido descomprimido en {DOWNLOAD_DIR}, se omite la descarga.")
        return DOWNLOAD_DIR

    api = KaggleApi()
    api.authenticate()

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    zips = list(DOWNLOAD_DIR.glob("*.zip"))
    if not zips:
        print(f"Descargando {KAGGLE_DATASET_SLUG} ...")
        api.dataset_download_files(KAGGLE_DATASET_SLUG, path=str(DOWNLOAD_DIR), unzip=False)
        zips = list(DOWNLOAD_DIR.glob("*.zip"))

    if not zips:
        raise SystemExit("No se descargo ningun .zip. Revisa el slug del dataset en config.py")

    print(f"Descomprimiendo {zips[0].name} ...")
    with zipfile.ZipFile(zips[0]) as zf:
        zf.extractall(DOWNLOAD_DIR)

    return DOWNLOAD_DIR


def find_class_dirs(download_dir: Path) -> dict:
    """Busca, para cada clase, una carpeta cuyo nombre coincida EXACTAMENTE
    (ignorando mayusculas) con el nombre de la clase. Evita falsos positivos
    por substring (ej. 'stand' no debe matchear 'handstand')."""
    class_dirs = {}
    for path in download_dir.rglob("*"):
        if not path.is_dir():
            continue
        name = path.name.lower()
        for class_name in CLASSES:
            if name == class_name.lower() and class_name not in class_dirs:
                class_dirs[class_name] = path
    return class_dirs


def filter_classes(download_dir: Path) -> None:
    class_dirs = find_class_dirs(download_dir)
    found_counts = {c: 0 for c in CLASSES}

    for class_name in CLASSES:
        class_dir = class_dirs.get(class_name)
        if class_dir is None:
            continue

        dest_class_dir = RAW_DIR / class_name
        dest_class_dir.mkdir(parents=True, exist_ok=True)

        video_dirs = [p for p in class_dir.iterdir() if p.is_dir()]
        for video_dir in video_dirs:
            dest_video_dir = dest_class_dir / video_dir.name
            if dest_video_dir.exists():
                found_counts[class_name] += 1
                continue
            images = [p for p in video_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
            if not images:
                continue
            dest_video_dir.mkdir(parents=True, exist_ok=True)
            for img in images:
                shutil.copy2(img, dest_video_dir / img.name)
            found_counts[class_name] += 1

    print("Videos (carpetas de frames) copiados por clase:")
    for class_name, count in found_counts.items():
        print(f"  {class_name}: {count}")

    if any(count == 0 for count in found_counts.values()):
        print(
            "\nADVERTENCIA: alguna clase quedo en 0 videos. Revisa la "
            f"estructura descargada en {download_dir} y ajusta find_class_dirs() "
            "si la convencion de nombres/estructura es distinta."
        )


def main() -> None:
    check_credentials()
    download_dir = download_and_extract()
    filter_classes(download_dir)
    print(f"\nListo. Frames filtrados en: {RAW_DIR}")
    print(f"Puedes borrar la carpeta cruda si quieres liberar espacio: {download_dir}")


if __name__ == "__main__":
    main()
