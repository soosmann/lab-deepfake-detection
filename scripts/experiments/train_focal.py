import os, sys, torch, torch.nn as nn, glob
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from model_combined import CombinedDetector
from dataset_combined import CombinedDataset

BASE        = "/media/DiscoLocal/MUDLAVSP/e542476/Task 2 DeepFakes Detection-20260226/Task 2 DeepFakes Detection-20260226"
DATA_DIR    = os.path.join(BASE, "FaceForensics_Faces_1.3x")
MODEL_DIR   = os.path.join(BASE, "models")
TS_PATH     = os.path.join(BASE, "models/yermandy/model.torchscript")
EPOCHS      = 20
BATCH_SIZE  = 16
ACCUM_STEPS = 4
LR_ATTN     = 1e-4
LR_FC       = 3e-4
WEIGHT_DECAY= 1e-4
VAL_SPLIT   = 0.15
P_SBI       = 0.5
NUM_WORKERS = 4
MARGIN      = 0.3
GAMMA       = 2.0  # Focal loss gamma


class FocalPairedLoss(nn.Module):
    """Focal Loss + AUC margin loss.
    Focal loss da más peso a ejemplos difíciles (frames ambiguos),
    lo que mejora la generalización cross-dataset.
    """
    def __init__(self, margin=0.3, gamma=2.0):
        super().__init__()
        self.margin = margin
        self.gamma  = gamma

    def forward(self, preds_real, preds_fake):
        pr = torch.sigmoid(preds_real)
        pf = torch.sigmoid(preds_fake)

        # Focal loss — penaliza más los ejemplos difíciles
        focal_real = -((1 - pr) ** self.gamma) * torch.log(pr + 1e-8)
        focal_fake = -(pf ** self.gamma) * torch.log(1 - pf + 1e-8)
        loss_focal = (focal_real.mean() + focal_fake.mean()) / 2.0

        # AUC margin loss
        loss_auc = torch.mean(torch.relu(pf - pr + self.margin))

        return loss_focal + loss_auc


def find_last_checkpoint():
    ckpts = glob.glob(os.path.join(MODEL_DIR, "focal_epoch*.pth"))
    if not ckpts:
        return None, 0
    latest = max(ckpts, key=lambda p: int(p.split("epoch")[-1].split(".")[0]))
    return latest, int(latest.split("epoch")[-1].split(".")[0])


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("=" * 60)
    print("  Entrenamiento Focal Loss (CLIP + attention_did)")
    print(f"  Backbone : yermandy congelado")
    print(f"  Loss     : Focal (gamma={GAMMA}) + AUC margin")
    print(f"  Dispositivo: {device} | P(SBI): {P_SBI:.0%}")
    print("=" * 60)

    full_ds = CombinedDataset(DATA_DIR, augment=True, p_sbi=P_SBI)
    n_val   = int(len(full_ds) * VAL_SPLIT)
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))
    val_ds.dataset.augment = False

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    print(f"  Train: {n_train:,} | Val: {n_val:,} | Batches: {len(train_loader)}")

    model     = CombinedDetector(TS_PATH).to(device)
    criterion = FocalPairedLoss(MARGIN, GAMMA)

    optimizer = torch.optim.AdamW([
        {"params": model.attention_did.parameters(), "lr": LR_ATTN},
        {"params": model.fc.parameters(),            "lr": LR_FC},
    ], weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda")

    start_epoch = 0
    best_auc    = 0.0
    ckpt_path, last_epoch = find_last_checkpoint()
    if ckpt_path:
        print(f"  Resumiendo desde época {last_epoch}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.attention_did.load_state_dict(ckpt["attention_did"])
        model.fc.load_state_dict(ckpt["fc"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        best_auc    = ckpt.get("best_auc", 0.0)
        start_epoch = last_epoch
    else:
        print("  Entrenamiento desde cero.")

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        model.backbone.eval()
        running_loss = 0.0
        optimizer.zero_grad()

        for i, (real_imgs, fake_imgs, _, _) in enumerate(
                tqdm(train_loader, desc=f"Época {epoch+1}/{EPOCHS}")):

            real_imgs = real_imgs.to(device)
            fake_imgs = fake_imgs.to(device)

            with torch.amp.autocast("cuda"):
                loss = criterion(model(real_imgs), model(fake_imgs)) / ACCUM_STEPS

            scaler.scale(loss).backward()

            if (i + 1) % ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(model.attention_did.parameters()) +
                    list(model.fc.parameters()), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * ACCUM_STEPS

        scheduler.step()

        # Validación
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for real_imgs, fake_imgs, _, _ in val_loader:
                real_imgs = real_imgs.to(device)
                fake_imgs = fake_imgs.to(device)
                with torch.amp.autocast("cuda"):
                    pr = torch.sigmoid(model(real_imgs)).cpu().squeeze()
                    pf = torch.sigmoid(model(fake_imgs)).cpu().squeeze()
                all_probs.extend(pr.tolist() if pr.dim() > 0 else [pr.item()])
                all_probs.extend(pf.tolist() if pf.dim() > 0 else [pf.item()])
                all_labels.extend([1.0] * real_imgs.size(0))
                all_labels.extend([0.0] * fake_imgs.size(0))

        val_auc = roc_auc_score(all_labels, all_probs)
        print(f"Época [{epoch+1}/{EPOCHS}] | Loss: {running_loss/len(train_loader):.4f} | "
              f"Val AUC: {val_auc:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        ckpt_data = {
            "epoch":         epoch + 1,
            "attention_did": model.attention_did.state_dict(),
            "fc":            model.fc.state_dict(),
            "optimizer":     optimizer.state_dict(),
            "scheduler":     scheduler.state_dict(),
            "val_auc":       val_auc,
            "best_auc":      best_auc,
        }
        torch.save(ckpt_data, os.path.join(MODEL_DIR, f"focal_epoch{epoch+1}.pth"))

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(ckpt_data, os.path.join(MODEL_DIR, "best_focal.pth"))
            print(f"  ✅ Nuevo mejor modelo focal (AUC: {best_auc:.4f})")

    print(f"\n  Completado. Mejor Val AUC: {best_auc:.4f}")


if __name__ == "__main__":
    train()