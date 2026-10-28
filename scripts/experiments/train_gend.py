import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_gend   import LNCLIPDetector
from dataset_combined import CombinedDataset, get_base_transform
from loss_gend    import GenDLoss, slerp_augmentation

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────

BASE = (r"/media/DiscoLocal/MUDLAVSP/e542476/Task 2 DeepFakes Detection-20260226"
        r"/Task 2 DeepFakes Detection-20260226")

DATA_DIR   = os.path.join(BASE, "FaceForensics_Faces_1.3x")
MODELS_DIR = os.path.join(BASE, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_OUT  = os.path.join(MODELS_DIR, "best_gend.pth")
PLOT_OUT   = os.path.join(BASE, "gend_training.png")

# Hiperparámetros (siguiendo el paper GenD)
EPOCHS        = 30
BATCH_SIZE    = 16
ACCUM_STEPS   = 4        # efectivo batch size = 64       # paper usa 128 — reducimos si hay OOM
LR_LN         = 1e-4    # LN params
LR_CLS        = 5e-4    # clasificador lineal
WEIGHT_DECAY  = 1e-4
VAL_SPLIT     = 0.15
MAX_PER_VIDEO = 32       # frames por vídeo para balancear

# Pesos de la loss (paper GenD)
LAMBDA_UNIF  = 0.5
LAMBDA_ALIGN = 1.0

# Slerp
USE_SLERP    = True
SLERP_T_MIN  = 0.3
SLERP_T_MAX  = 0.7

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# VALIDACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_auc(model, loader, device):
    model.eval()
    all_labels = []
    all_probs  = []

    with torch.no_grad(), autocast('cuda'):
        for real_imgs, fake_imgs, _, _ in loader:
            real_imgs = real_imgs.to(device)
            fake_imgs = fake_imgs.to(device)

            logits_real, _ = model(real_imgs)
            logits_fake, _ = model(fake_imgs)

            probs_real = torch.sigmoid(logits_real).squeeze(-1).cpu().tolist()
            probs_fake = torch.sigmoid(logits_fake).squeeze(-1).cpu().tolist()

            if isinstance(probs_real, float):
                probs_real = [probs_real]
                probs_fake = [probs_fake]

            all_probs  += probs_real + probs_fake
            all_labels += [1.0] * len(probs_real) + [0.0] * len(probs_fake)

    return roc_auc_score(all_labels, all_probs)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────────────────────

def train():
    print("=" * 60)
    print("  Entrenamiento LNCLIP-DF / GenD")
    print(f"  Dispositivo  : {DEVICE}")
    print(f"  Epochs       : {EPOCHS}")
    print(f"  Batch size   : {BATCH_SIZE}")
    print(f"  Slerp        : {USE_SLERP}")
    print("=" * 60)

    # ── Dataset ──────────────────────────────────────────────────────────────
    full_ds = CombinedDataset(
        root_dir=DATA_DIR,
        augment=True,
        p_sbi=0.5,           # 50% SBI + 50% FF++ paired
        max_per_video=MAX_PER_VIDEO,
    )

    val_size   = int(len(full_ds) * VAL_SPLIT)
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(
        full_ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Val sin augmentación
    val_ds.dataset.augment   = False
    val_ds.dataset.transform = get_base_transform()

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=4,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=4,
                              pin_memory=True)

    print(f"  Train: {train_size:,} pares | Val: {val_size:,} pares")
    print(f"  Batches por época: {len(train_loader)}")

    # ── Modelo ───────────────────────────────────────────────────────────────
    model = LNCLIPDetector().to(DEVICE)

    # ── Optimizador ──────────────────────────────────────────────────────────
    # Separar LN params del clasificador para distintos LR
    ln_params  = []
    cls_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'classifier' in name:
            cls_params.append(p)
        else:
            ln_params.append(p)

    optimizer = torch.optim.AdamW([
        {'params': ln_params,  'lr': LR_LN},
        {'params': cls_params, 'lr': LR_CLS},
    ], weight_decay=WEIGHT_DECAY)

    # Cosine annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    criterion = GenDLoss(
        lambda_unif=LAMBDA_UNIF,
        lambda_align=LAMBDA_ALIGN,
    )
    scaler = GradScaler('cuda')

    # ── Resume desde último checkpoint ───────────────────────────────────────
    best_auc    = 0.0
    start_epoch = 1
    history     = {"train_loss": [], "val_auc": []}

    # Buscar el checkpoint más reciente
    existing = sorted([
        f for f in os.listdir(MODELS_DIR)
        if f.startswith("gend_epoch") and f.endswith(".pth")
    ], key=lambda x: int(x.replace("gend_epoch","").replace(".pth","")))

    if existing:
        last_ckpt = os.path.join(MODELS_DIR, existing[-1])
        print(f"▶ Resumiendo desde: {last_ckpt}")
        ckpt = torch.load(last_ckpt, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_auc    = ckpt["val_auc"]
        for _ in range(ckpt["epoch"]):
            scheduler.step()
        print(f"  ▶ Continuando desde época {start_epoch} | Mejor AUC: {best_auc:.4f}")

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        optimizer.zero_grad()
        for batch_idx, (real_imgs, fake_imgs, _, _) in enumerate(train_loader, 1):
            real_imgs = real_imgs.to(DEVICE)
            fake_imgs = fake_imgs.to(DEVICE)

            # Construir batch mixto: [real, fake] → labels [1, 0]
            imgs   = torch.cat([real_imgs, fake_imgs], dim=0)
            labels = torch.cat([
                torch.ones(real_imgs.size(0)),
                torch.zeros(fake_imgs.size(0))
            ]).to(DEVICE)

            with autocast('cuda'):
                logits, features = model(imgs)

                # Slerp augmentation en espacio latente
                if USE_SLERP:
                    aug_features, aug_labels = slerp_augmentation(
                        features, labels,
                        t_range=(SLERP_T_MIN, SLERP_T_MAX)
                    )
                    aug_logits = model.classifier(aug_features)
                    all_logits   = torch.cat([logits, aug_logits], dim=0)
                    all_features = torch.cat([features, aug_features], dim=0)
                    all_labels   = torch.cat([labels, aug_labels], dim=0)
                else:
                    all_logits   = logits
                    all_features = features
                    all_labels   = labels

                loss, components = criterion(all_logits, all_features, all_labels)
                loss = loss / ACCUM_STEPS  # normalizar por acumulación

            scaler.scale(loss).backward()

            if batch_idx % ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += loss.item() * ACCUM_STEPS  # recuperar escala real

            if batch_idx % 50 == 0:
                print(f"  Época {epoch}/{EPOCHS} | "
                      f"Batch {batch_idx}/{len(train_loader)} | "
                      f"Loss: {loss.item()*ACCUM_STEPS:.4f} "
                      f"(bce={components['bce']:.3f} "
                      f"unif={components['unif']:.3f} "
                      f"align={components['align']:.3f})")

        scheduler.step()

        # Validación
        val_auc = evaluate_auc(model, val_loader, DEVICE)
        avg_loss = epoch_loss / len(train_loader)
        lr_now   = optimizer.param_groups[0]['lr']

        print(f"Época [{epoch}/{EPOCHS}] | "
              f"Train Loss: {avg_loss:.4f} | "
              f"Val AUC: {val_auc:.4f} ({100*val_auc:.2f}%) | "
              f"LR: {lr_now:.2e}")

        history["train_loss"].append(avg_loss)
        history["val_auc"].append(val_auc)

        # Guardar checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_auc': val_auc,
        }
        torch.save(checkpoint,
                   os.path.join(MODELS_DIR, f"gend_epoch{epoch}.pth"))

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(checkpoint, MODEL_OUT)
            print(f"  ✅ Nuevo mejor modelo guardado (AUC: {best_auc:.4f})")

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_loss"]); ax1.set_title("Train Loss")
    ax2.plot(history["val_auc"]);    ax2.set_title("Val AUC")
    plt.tight_layout()
    plt.savefig(PLOT_OUT, dpi=100)
    plt.close()

    print(f"\n{'='*60}")
    print(f"  Entrenamiento completado.")
    print(f"  Mejor Val AUC : {best_auc:.4f} ({100*best_auc:.2f}%)")
    print(f"  Modelo guardado en: {MODEL_OUT}")
    print(f"{'='*60}")


if __name__ == "__main__":
    train()