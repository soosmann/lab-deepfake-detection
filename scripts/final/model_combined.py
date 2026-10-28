import torch
import torch.nn as nn

# ─────────────────────────────────────────────────────────────────────────────
# CombinedDetector
#
# Backbone : yermandy/deepfake-detection (TorchScript)
#            CLIP ViT-L/14 + LN-tuning entrenado con 4 métodos FF++
#            Completamente CONGELADO — devuelve features L2 [B, 1024]
#
# Cabeza   : attention_did + fc (entrenable)
#            Self-attention que amplifica artefactos de forgery
#            Dropout 0.5 para regularización
# ─────────────────────────────────────────────────────────────────────────────

class CombinedDetector(nn.Module):
    def __init__(self, torchscript_path, embed_dim=1024):
        super().__init__()

        print(f"[CombinedDetector] Cargando backbone desde {torchscript_path}")
        self.backbone = torch.jit.load(torchscript_path, map_location='cpu')
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        # attention_did: self-attention sobre features 1024-dim
        self.attention_did = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=16, batch_first=True
        )

        # Clasificador
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

        total = sum(p.numel() for p in self.parameters())
        train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  embed_dim          : {embed_dim}")
        print(f"  Params entrenables : {train:,} / {total:,} ({100*train/total:.2f}%)")
        print(f"  [OK] Modelo listo.")

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone.feature_extractor(x.float())  # [B, 1024]

        features = features.unsqueeze(1)  # [B, 1, 1024]
        attn_out, _ = self.attention_did(features, features, features)
        attn_out = attn_out.squeeze(1)    # [B, 1024]
        return self.fc(attn_out)


if __name__ == "__main__":
    import os
    BASE = "/media/DiscoLocal/MUDLAVSP/e542476/Task 2 DeepFakes Detection-20260226/Task 2 DeepFakes Detection-20260226"
    path = os.path.join(BASE, "models/yermandy/model.torchscript")
    model = CombinedDetector(path)
    x = torch.randn(4, 3, 224, 224)
    out = model(x)
    print(f"output shape: {out.shape}")
    print("✅ model_combined.py OK")