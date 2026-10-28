import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# FreqDetector
#
# Backbone CLIP  : yermandy/deepfake-detection (congelado) → [B, 1024]
#                  LN-tuning entrenado con 4 métodos FF++
#
# Rama frecuencia: DWT (Discrete Wavelet Transform) → CNN → [B, 256]
#                  Captura artefactos en altas frecuencias que CLIP ignora:
#                  bordes de blending, inconsistencias GAN, ringing artifacts
#
# Fusión         : concat [B, 1280] → attention_did → fc
# ─────────────────────────────────────────────────────────────────────────────

class DWTBranch(nn.Module):
    """
    Rama de frecuencia con Haar DWT manual.
    Descompone la imagen en 4 subbandas: LL, LH, HL, HH
    Las subbandas de alta frecuencia (LH, HL, HH) contienen artefactos de forgery.
    """
    def __init__(self, out_dim=256):
        super().__init__()
        # Haar wavelet filters
        self.register_buffer('haar_ll', torch.tensor([[1,1],[1,1]], dtype=torch.float32) / 4.0)
        self.register_buffer('haar_lh', torch.tensor([[-1,-1],[1,1]], dtype=torch.float32) / 4.0)
        self.register_buffer('haar_hl', torch.tensor([[-1,1],[-1,1]], dtype=torch.float32) / 4.0)
        self.register_buffer('haar_hh', torch.tensor([[1,-1],[-1,1]], dtype=torch.float32) / 4.0)

        # CNN sobre las 12 subbandas (3 canales × 4 subbandas)
        self.cnn = nn.Sequential(
            nn.Conv2d(12, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(256, out_dim)

    def dwt2d(self, x):
        """Haar DWT 2D sobre tensor [B, C, H, W]"""
        B, C, H, W = x.shape
        # Reshape para aplicar filtros por canal
        x_pad = F.pad(x, (0, W%2, 0, H%2), mode='reflect')
        # Unfold para aplicar filtros 2x2
        x_unf = x_pad.unfold(2, 2, 2).unfold(3, 2, 2)  # [B, C, H//2, W//2, 2, 2]
        x_unf = x_unf.contiguous().view(B, C, -1, 4)    # [B, C, H//2*W//2, 4]

        filters = torch.stack([
            self.haar_ll.view(-1),
            self.haar_lh.view(-1),
            self.haar_hl.view(-1),
            self.haar_hh.view(-1),
        ], dim=0)  # [4, 4]

        subbands = x_unf @ filters.T  # [B, C, H//2*W//2, 4]
        h, w = x_pad.shape[2]//2, x_pad.shape[3]//2
        subbands = subbands.view(B, C, h, w, 4)
        subbands = subbands.permute(0, 1, 4, 2, 3)  # [B, C, 4, H//2, W//2]
        return subbands.reshape(B, C*4, h, w)        # [B, 4C, H//2, W//2]

    def forward(self, x):
        # x: [B, 3, 224, 224] — imagen normalizada
        freq = self.dwt2d(x)   # [B, 12, 112, 112]
        feat = self.cnn(freq)  # [B, 256, 1, 1]
        feat = feat.squeeze(-1).squeeze(-1)  # [B, 256]
        return self.proj(feat)  # [B, out_dim]


class FreqDetector(nn.Module):
    def __init__(self, torchscript_path, clip_dim=1024, freq_dim=256):
        super().__init__()
        fused_dim = clip_dim + freq_dim

        print(f"[FreqDetector] Cargando backbone desde {torchscript_path}")
        self.backbone = torch.jit.load(torchscript_path, map_location='cpu')
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Rama de frecuencia (entrenable)
        self.freq_branch = DWTBranch(out_dim=freq_dim)

        # attention_did sobre features fusionadas
        self.attention_did = nn.MultiheadAttention(
            embed_dim=fused_dim, num_heads=16, batch_first=True
        )

        # Clasificador
        self.fc = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

        total = sum(p.numel() for p in self.parameters())
        train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  clip_dim + freq_dim : {clip_dim} + {freq_dim} = {fused_dim}")
        print(f"  Params entrenables  : {train:,} / {total:,} ({100*train/total:.2f}%)")
        print(f"  [OK] Modelo listo.")

    def forward(self, x):
        # CLIP features (backbone congelado)
        with torch.no_grad():
            clip_feat = self.backbone.feature_extractor(x.float())  # [B, 1024]

        # Frecuencia features (entrenable)
        freq_feat = self.freq_branch(x)  # [B, 256]

        # Fusión
        fused = torch.cat([clip_feat, freq_feat], dim=1)  # [B, 1280]

        # attention_did
        fused = fused.unsqueeze(1)
        attn_out, _ = self.attention_did(fused, fused, fused)
        attn_out = attn_out.squeeze(1)  # [B, 1280]

        return self.fc(attn_out)


if __name__ == "__main__":
    import os
    BASE = "/media/DiscoLocal/MUDLAVSP/e542476/Task 2 DeepFakes Detection-20260226/Task 2 DeepFakes Detection-20260226"
    path = os.path.join(BASE, "models/yermandy/model.torchscript")
    model = FreqDetector(path)
    x = torch.randn(4, 3, 224, 224)
    out = model(x)
    print(f"output shape: {out.shape}")
    print("✅ model_freq.py OK")