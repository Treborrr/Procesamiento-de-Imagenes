"""Entrena y evalua el modelo profundo: ResNet50 pre-entrenado en ImageNet
con transfer learning (seccion 3.3 del informe).

Pipeline: congelar conv1..layer3 -> afinar (fine-tune) el ultimo bloque
residual (layer4) + una nueva capa FC de 3 clases, con Adam (lr=0.001)
durante RESNET_EPOCHS epocas, con data augmentation (rotacion +/-15
grados, flip horizontal, zoom +/-10%) y perdida ponderada por clase
(class_weight) para compensar el desbalance de videos por clase.

Nota de rendimiento (CPU, sin GPU): conv1..layer3 no se re-entrenan, asi
que su salida para una misma imagen de entrada es siempre la misma. Se
cachea esa salida (mapa de caracteristicas 1024x14x14) UNA sola vez por
imagen -- evitando recalcular la mayor parte del costo de ResNet50 en
cada epoca -- y solo layer4+FC (que si se entrenan) corren con forward+
backward real en cada epoca sobre ese cache. Ver DECISIONES_TECNICAS.md,
seccion 8 y 12.

Uso:
    python train_resnet50.py
"""

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

from config import (
    CLASSES,
    FRAMES_DIR,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MODELS_DIR,
    RESNET_BATCH_SIZE,
    RESNET_EPOCHS,
    RESNET_IMAGE_SIZE,
    RESNET_LR,
    SPLITS_DIR,
)
from eval_utils import report, report_video_level

MODEL_NAME = "resnet50"
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}

train_transform = transforms.Compose([
    transforms.Resize(RESNET_IMAGE_SIZE),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(RESNET_IMAGE_SIZE, scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_transform = transforms.Compose([
    transforms.Resize(RESNET_IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class FramesDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform):
        self.frame_paths = df["frame_path"].tolist()
        self.labels = [CLASS_TO_IDX[c] for c in df["class"]]
        self.transform = transform

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        img = Image.open(FRAMES_DIR / self.frame_paths[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]


class FrozenPrefix(nn.Module):
    """conv1..layer3 de ResNet50, congelado. Salida: mapa de 1024x14x14
    (para entrada de 224x224) que alimenta a layer4."""

    def __init__(self, resnet: nn.Module):
        super().__init__()
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        return self.layer3(x)


class FineTuneHead(nn.Module):
    """layer4 + avgpool + FC de 3 clases. Esta es la parte que SI se
    entrena (fine-tuning real, no solo un clasificador lineal)."""

    def __init__(self, resnet: nn.Module, num_classes: int):
        super().__init__()
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self.fc = nn.Linear(resnet.fc.in_features, num_classes)

    def forward(self, x):
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


@torch.no_grad()
def extract_prefix_features(prefix, loader, device, desc) -> tuple[torch.Tensor, np.ndarray]:
    """Cachea la salida de la parte congelada, en float16 para ahorrar RAM
    (~5GB para todo el train+test en vez de ~10GB en float32)."""
    feats, labels = [], []
    for images, batch_labels in tqdm(loader, desc=desc):
        images = images.to(device)
        feats.append(prefix(images).half().cpu())
        labels.append(batch_labels.numpy())
    return torch.cat(feats), np.concatenate(labels)


def train_finetune_head(X_train: torch.Tensor, y_train: np.ndarray, device) -> nn.Module:
    base_resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    head = FineTuneHead(base_resnet, len(CLASSES)).to(device)

    class_weights = compute_class_weight("balanced", classes=np.arange(len(CLASSES)), y=y_train)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    optimizer = optim.Adam(head.parameters(), lr=RESNET_LR)

    y = torch.tensor(y_train, dtype=torch.long)

    head.train()
    for epoch in range(RESNET_EPOCHS):
        permutation = torch.randperm(X_train.size(0))
        running_loss, correct = 0.0, 0
        for i in tqdm(range(0, X_train.size(0), RESNET_BATCH_SIZE), desc=f"Fine-tuning epoca {epoch + 1}/{RESNET_EPOCHS}"):
            idx = permutation[i:i + RESNET_BATCH_SIZE]
            # Upcast a float32 solo para el minibatch actual (estabilidad numerica en CPU).
            batch_x = X_train[idx].float().to(device)
            batch_y = y[idx].to(device)

            optimizer.zero_grad()
            outputs = head(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)
            correct += (outputs.argmax(1) == batch_y).sum().item()

        print(f"Epoca {epoch + 1}/{RESNET_EPOCHS}  loss={running_loss / X_train.size(0):.4f}  "
              f"train_acc={correct / X_train.size(0):.4f}")

    return head


@torch.no_grad()
def evaluate_head(head, X_test: torch.Tensor, y_test: np.ndarray, device):
    head.eval()
    preds = []
    for i in range(0, X_test.size(0), RESNET_BATCH_SIZE):
        batch_x = X_test[i:i + RESNET_BATCH_SIZE].float().to(device)
        preds.append(head(batch_x).argmax(1).cpu())
    preds = torch.cat(preds).numpy()
    y_true = [IDX_TO_CLASS[i] for i in y_test]
    y_pred = [IDX_TO_CLASS[i] for i in preds]
    return y_true, y_pred


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando device: {device}")

    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    test_df = pd.read_csv(SPLITS_DIR / "test.csv")

    prefix = FrozenPrefix(models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)).to(device)

    # Features de train: version original (eval_transform) + una version
    # aumentada (train_transform), para conservar el efecto de data
    # augmentation sin recalcular la parte congelada en cada epoca.
    train_loader_plain = DataLoader(
        FramesDataset(train_df, eval_transform), batch_size=RESNET_BATCH_SIZE, num_workers=0,
    )
    train_loader_aug = DataLoader(
        FramesDataset(train_df, train_transform), batch_size=RESNET_BATCH_SIZE, num_workers=0,
    )
    test_loader = DataLoader(
        FramesDataset(test_df, eval_transform), batch_size=RESNET_BATCH_SIZE, num_workers=0,
    )

    X_plain, y_plain = extract_prefix_features(prefix, train_loader_plain, device, "Features train (original)")
    X_aug, y_aug = extract_prefix_features(prefix, train_loader_aug, device, "Features train (augmented)")
    X_train = torch.cat([X_plain, X_aug])
    y_train = np.concatenate([y_plain, y_aug])

    X_test, y_test = extract_prefix_features(prefix, test_loader, device, "Features test")
    video_ids_test = test_df["video_id"].tolist()

    print(f"\nFeatures de entrenamiento: {tuple(X_train.shape)}  (incluye original + augmented, float16)")

    head = train_finetune_head(X_train, y_train, device)
    y_true, y_pred = evaluate_head(head, X_test, y_test, device)

    print("\n=== Resultados ResNet50 (fine-tuning layer4, por frame) ===")
    report(y_true, y_pred, MODEL_NAME)
    report_video_level(video_ids_test, y_true, y_pred, MODEL_NAME)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    full_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    full_model.layer4.load_state_dict(head.layer4.state_dict())
    full_model.fc = head.fc
    torch.save(full_model.state_dict(), MODELS_DIR / "resnet50.pt")
    print(f"\nModelo guardado en: {MODELS_DIR / 'resnet50.pt'}")


if __name__ == "__main__":
    main()
