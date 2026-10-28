import os, sys, torch, torch.nn as nn, torch.nn.functional as F, glob
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
LAMBDA_SC   = 0.5   # peso de SupCon vs BCE
TEMP_SC     = 0.07  # temperatura SupCon (paper original usa 0.07)


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).
    Aprende que features de la misma clase estén juntas en la hiperesfera
    y features de clases distintas separadas.
    Esto generaliza mejor cross-dataset que BCE puro.
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        # features: [B, D] — ya L2-normalizadas por el backbone
        # labels  : [B]   — 1=real, 0=fake
        device = features.device
        B = features.shape[0]

        # L2 normalizar features
        features = F.normalize(features, dim=1)

        # Matriz de similitud coseno [B, B]
        sim = torch.matmul(features, features.T) / self.temperature

        # Máscara de positivos (misma clase, distinto ejemplo)
        labels = labels.view(-1, 1)
        mask_pos = (labels == labels.T).float().to(device)
        mask_pos.fill_diagonal_(0)  # excluir el propio ejemplo

        # Log-softmax sobre todos los pares excepto diagonal
        mask_diag = torch.eye(B, dtype=torch.bool, device=device)
        sim.masked_fill_(mask_diag, float('-inf'))

        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)

        # Media sobre positivos
        n_pos = mask_pos.sum(dim=1)
        n_pos = torch.clamp(n_pos, min=1)
        loss = -(mask_pos * log_prob).sum(dim=1) / n_pos

        return loss.mean()


class SupConPairedLoss(nn.Module):
    """BCE + AUC margin + SupCon"""
    def __init__(self, margin=0.3, lambda_sc=0.5, temp_sc=0.07):
        super().__init__()
        self.bce       = nn.BCEWithLogitsLoss()
        self.supcon    = SupConLoss(temperature=temp_sc)
        self.margin    = margin
        self.lambda_sc = lambda_sc

    def forward(self, preds_real, preds_fake, feats_real, feats_fake):
        # BCE
        loss_bce = (self.bce(preds_real, torch.ones_like(preds_real)) +
                    self.bce(preds_fake, torch.zeros_like(preds_fake))) / 2.0

        # AUC margin
        pr = torch.sigmoid(preds_real)
        pf = torch.sigmoid(preds_fake)
        loss_auc = torch.mean(torch.relu(pf - pr + self.margin))

        # SupCon — concatenar features y labels
        features = torch.cat([feats_real, feats_fake], dim=0)
        labels   = torch.cat([
            torch.ones(feats_real.shape[0], device=feats_real.device),
            torch.zeros(feats_fake.shape[0], device=feats_fake.device)
        ])
        loss_sc = self.supcon(features, labels)

        return loss_bce + loss_auc + self.lambda_sc * loss_sc


# ── Modelo extendido que devuelve features + logits ──────────────────────────
class CombinedDetectorWithFeatures(CombinedDetector):
    def forward_with_features(self, x):
        with torch.no_grad():
            clip_feat = self.backbone.feature_extractor(x.float())
        clip_feat = clip_feat.unsqueeze(1)
        attn_out, _ = self.attention_did(clip_feat, clip_feat, clip_feat)
        attn_out = attn_out.squeeze(1)
        logits = self.fc(attn_out)
        return logits, attn_out  # devuelve también features pre-clasificador


def find_last_checkpoint():
    ckpts = glob.glob(os.path.join(MODEL_DIR, "supcon_epoch*.pth"))
    if not ckpts:
        return None, 0
    latest = max(ckpts, key=lambda p: int(p.split("epoch")[-1].split(".")[0]))
    return latest, int(latest.split("epoch")[-1].split(".")[0])


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("=" * 60)
    print("  Entrenamiento SupCon (BCE + AUC margin + SupCon)")
    print(f"  Backbone   : yermandy congelado")
    print(f"  Lambda SupCon: {LAMBDA_SC} | Temp: {TEMP_SC}")
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

    model     = CombinedDetectorWithFeatures(TS_PATH).to(device)
    criterion = SupConPairedLoss(MARGIN, LAMBDA_SC, TEMP_SC)

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
                pr, fr = model.forward_with_features(real_imgs)
                pf, ff = model.forward_with_features(fake_imgs)
                loss = criterion(pr, pf, fr, ff) / ACCUM_STEPS

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
                    pr, _ = model.forward_with_features(real_imgs)
                    pf, _ = model.forward_with_features(fake_imgs)
                    pr = torch.sigmoid(pr).cpu().squeeze()
                    pf = torch.sigmoid(pf).cpu().squeeze()
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
        torch.save(ckpt_data, os.path.join(MODEL_DIR, f"supcon_epoch{epoch+1}.pth"))

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(ckpt_data, os.path.join(MODEL_DIR, "best_supcon.pth"))
            print(f"  ✅ Nuevo mejor modelo supcon (AUC: {best_auc:.4f})")

    print(f"\n  Completado. Mejor Val AUC: {best_auc:.4f}")


if __name__ == "__main__":
    train()