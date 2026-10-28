import torch
import torch.nn as nn
import clip

# ─────────────────────────────────────────────────────────────────────────────
# LNCLIP-DF / GenD  — Implementación propia basada en:
#   "Unlocking the Hidden Potential of CLIP in Generalizable Deepfake Detection"
#   "Deepfake Detection that Generalizes Across Benchmarks" (WACV 2026)
#
# Arquitectura exacta del paper:
#   1. CLIP ViT-L/14  — backbone congelado
#   2. LN-tuning      — solo LayerNorm params entrenables (0.03% del total)
#   3. L2 norm        — normalizar CLS token a hiperesfera unitaria
#   4. Linear         — clasificador lineal (1024 → 1)
#
# Lo que NO hace el paper (y nosotros añadimos):
#   + Slerp se aplica en el training loop (ver train_gend.py)
#   + UnAl loss       — uniformity + alignment en hiperesfera
# ─────────────────────────────────────────────────────────────────────────────

class LNCLIPDetector(nn.Module):
    """
    CLIP ViT-L/14 con LN-tuning + L2 norm + clasificador lineal.

    Solo los parámetros de LayerNorm son entrenables.
    El CLS token se L2-normaliza antes del clasificador.
    """

    def __init__(self):
        super().__init__()

        print("[LNCLIPDetector] Cargando CLIP ViT-L/14...")
        clip_model, _ = clip.load("ViT-L/14", device="cpu")
        self.encoder = clip_model.visual
        self.encoder = self.encoder.float()  # fp32 para estabilidad

        # ── Congelar todo ──────────────────────────────────────────────────
        for p in self.encoder.parameters():
            p.requires_grad = False

        # ── LN-tuning: descongelar solo LayerNorm ─────────────────────────
        # Según el paper: 0.03% del total de parámetros
        n_ln = 0
        for name, module in self.encoder.named_modules():
            if isinstance(module, nn.LayerNorm):
                for p in module.parameters():
                    p.requires_grad = True
                n_ln += 1

        # ── Clasificador lineal ────────────────────────────────────────────
        embed_dim = self.encoder.output_dim   # 1024 para ViT-L/14
        self.classifier = nn.Linear(embed_dim, 1)

        # Stats
        total  = sum(p.numel() for p in self.parameters())
        train  = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  LN blocks entrenables : {n_ln}")
        print(f"  embed_dim             : {embed_dim}")
        print(f"  Params entrenables    : {train:,} / {total:,} ({100*train/total:.2f}%)")
        print(f"  [OK] Modelo listo.")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extrae el CLS token de CLIP y lo L2-normaliza.
        Retorna features en la hiperesfera unitaria.
        shape: [B, 1024]
        """
        features = self.encoder(x)           # [B, 1024]
        features = nn.functional.normalize(features, p=2, dim=-1)  # L2 norm
        return features

    def forward(self, x: torch.Tensor):
        """
        Retorna (logits, features_l2).
        logits   : [B, 1]  para BCE
        features : [B, 1024]  para UnAl loss y slerp
        """
        features = self.encode(x)
        logits   = self.classifier(features)
        return logits, features


# ─────────────────────────────────────────────────────────────────────────────
# Test rápido
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = LNCLIPDetector()
    x = torch.randn(4, 3, 224, 224)
    logits, feats = model(x)
    print(f"logits shape : {logits.shape}")   # [4, 1]
    print(f"feats shape  : {feats.shape}")    # [4, 1024]
    print(f"feats norm   : {feats.norm(dim=-1)}")  # todos ~1.0
    print("✅ model_gend.py OK")