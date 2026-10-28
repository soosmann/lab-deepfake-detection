import os
import re
import cv2
import random
import numpy as np
from collections import defaultdict
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

# ─────────────────────────────────────────────────────────────────────────────
# Dataset Combinado: FF++ Paired + SBI
#
# Problema raíz: solo tenemos un método de fake (Deepfakes de FF++).
# El modelo aprende artefactos específicos de ese método y no generaliza.
#
# Solución: mezclar dos fuentes de fakes en cada época:
#   1. FF++ paired real     → fake real del mismo vídeo fuente
#      - Señal de paired training (esencial según GenD)
#      - Artefactos específicos del método Deepfakes
#
#   2. SBI (Self-Blended Images) desde los reales
#      - Artefactos de blending genéricos (independientes del método)
#      - Fuerza al modelo a aprender señales universales de forgery
#      - Compensa no tener F2F, FS, NT
#
# Cada __getitem__ devuelve con probabilidad p_sbi un par SBI,
# y con probabilidad (1-p_sbi) un par FF++ real.
# ─────────────────────────────────────────────────────────────────────────────

CLIP_MEAN = [0.48145466, 0.4578275,  0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]
IMG_SIZE  = 224


def get_base_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


def get_augment_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1,
                               saturation=0.05, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


class CombinedDataset(Dataset):
    """
    Dataset combinado FF++ paired + SBI.

    Args:
        root_dir     : carpeta con subcarpetas Real/ y Fake/
        augment      : aplicar augmentación de color
        p_sbi        : probabilidad de usar SBI en lugar de FF++ fake (0.0-1.0)
        max_per_video: frames máximos por vídeo
    """

    def __init__(self, root_dir, augment=True, p_sbi=0.5, max_per_video=32):
        self.augment       = augment
        self.p_sbi         = p_sbi
        self.max_per_video = max_per_video
        self.transform     = get_augment_transform() if augment else get_base_transform()

        real_dir = os.path.join(root_dir, "Real")
        fake_dir = os.path.join(root_dir, "Fake")

        # ── Indexar reales por vídeo ───────────────────────────────────────
        real_by_video = defaultdict(list)
        for f in sorted(os.listdir(real_dir)):
            if not f.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue
            m = re.match(r'^(\d+)_frame', f)
            vid = m.group(1) if m else f.split('_')[0]
            real_by_video[vid].append(os.path.join(real_dir, f))

        # ── Indexar fakes por vídeo fuente ────────────────────────────────
        fake_by_video = defaultdict(list)
        for f in sorted(os.listdir(fake_dir)):
            if not f.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue
            m = re.match(r'^(\d+)_\d+_frame', f)
            vid = m.group(1) if m else f.split('_')[0]
            fake_by_video[vid].append(os.path.join(fake_dir, f))

        # ── Construir pares FF++ ───────────────────────────────────────────
        common_videos = sorted(set(real_by_video.keys()) & set(fake_by_video.keys()))
        self.ff_pairs = []
        self.all_reals = []  # para SBI

        for vid in common_videos:
            reals = real_by_video[vid][:max_per_video]
            fakes = fake_by_video[vid][:max_per_video]
            n = min(len(reals), len(fakes))
            for i in range(n):
                self.ff_pairs.append((reals[i], fakes[i]))
            self.all_reals.extend(reals)

        # Añadir reales sin pareja para SBI
        for vid in real_by_video:
            if vid not in fake_by_video:
                self.all_reals.extend(real_by_video[vid][:max_per_video])

        print(f"[CombinedDataset] Vídeos paired FF++ : {len(common_videos)}")
        print(f"[CombinedDataset] Pares FF++         : {len(self.ff_pairs):,}")
        print(f"[CombinedDataset] Reales para SBI    : {len(self.all_reals):,}")
        print(f"[CombinedDataset] P(SBI)             : {p_sbi:.0%}")
        print(f"[CombinedDataset] Augment            : {augment}")

    # ─────────────────────────────────────────────────────────────────────────
    # SBI: Self-Blended Image
    # ─────────────────────────────────────────────────────────────────────────
    def _generate_sbi(self, img_np: np.ndarray):
        """Genera fake sintético mezclando la imagen consigo misma."""
        # Fuente: misma imagen con distinta compresión JPEG
        quality = random.randint(70, 98)
        _, enc  = cv2.imencode('.jpg', img_np.copy(),
                               [cv2.IMWRITE_JPEG_QUALITY, quality])
        source  = cv2.imdecode(enc, cv2.IMREAD_COLOR)

        # Máscara de blending con formas aleatorias
        h, w   = img_np.shape[:2]
        mask   = np.zeros((h, w), dtype=np.float32)
        cx, cy = w // 2, h // 2

        for _ in range(random.randint(1, 4)):
            if random.random() > 0.5:
                rx = random.randint(w // 8, w // 3)
                ry = random.randint(h // 8, h // 3)
                ox = random.randint(-w // 5, w // 5)
                oy = random.randint(-h // 5, h // 5)
                cv2.ellipse(mask, (cx + ox, cy + oy), (rx, ry),
                            random.uniform(0, 360), 0, 360, 1.0, -1)
            else:
                n_pts = random.randint(4, 8)
                pts = []
                for i in range(n_pts):
                    angle = 2 * np.pi * i / n_pts + random.uniform(-0.4, 0.4)
                    r = random.randint(min(w, h) // 8, min(w, h) // 3)
                    ox = random.randint(-w // 5, w // 5)
                    oy = random.randint(-h // 5, h // 5)
                    pts.append([int(cx + ox + r * np.cos(angle)),
                                int(cy + oy + r * np.sin(angle))])
                cv2.fillConvexPoly(mask, np.array(pts, dtype=np.int32), 1.0)

        ks   = random.choice([21, 31, 41, 51])
        mask = cv2.GaussianBlur(mask, (ks, ks), 0)
        mask = np.clip(mask, 0, 1)
        m3   = np.stack([mask] * 3, axis=2)

        fake = m3 * source + (1 - m3) * img_np
        fake = np.clip(fake, 0, 255).astype(np.uint8)

        # Ruido gaussiano leve
        sigma = random.uniform(0.0, 2.0)
        noise = np.random.normal(0, sigma, fake.shape).astype(np.float32)
        fake  = np.clip(fake.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return fake

    # ─────────────────────────────────────────────────────────────────────────
    # Dataset interface
    # ─────────────────────────────────────────────────────────────────────────
    def _to_tensor(self, img_np: np.ndarray) -> torch.Tensor:
        pil = Image.fromarray(img_np)
        if self.augment:
            return get_augment_transform()(pil)
        return get_base_transform()(pil)

    def __len__(self):
        return len(self.ff_pairs)

    def __getitem__(self, idx):
        # Flip sincronizado
        do_flip = self.augment and random.random() > 0.5

        if random.random() < self.p_sbi:
            # ── SBI: usar real aleatorio como base ────────────────────────
            real_path = random.choice(self.all_reals)
            real_img  = Image.open(real_path).convert("RGB")
            if do_flip:
                real_img = TF.hflip(real_img)

            real_np   = np.array(real_img.resize((IMG_SIZE, IMG_SIZE)))
            fake_np   = self._generate_sbi(real_np)

            real_t = self._to_tensor(real_np)
            fake_t = self._to_tensor(fake_np)
        else:
            # ── FF++ paired ───────────────────────────────────────────────
            real_path, fake_path = self.ff_pairs[idx]
            real_img = Image.open(real_path).convert("RGB")
            fake_img = Image.open(fake_path).convert("RGB")

            if do_flip:
                real_img = TF.hflip(real_img)
                fake_img = TF.hflip(fake_img)

            real_t = self.transform(real_img)
            fake_t = self.transform(fake_img)

        return (real_t, fake_t, torch.tensor(1.0), torch.tensor(0.0))


# ─────────────────────────────────────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ROOT = (r"/media/DiscoLocal/MUDLAVSP/e542476/Task 2 DeepFakes Detection-20260226"
            r"/Task 2 DeepFakes Detection-20260226/FaceForensics_Faces_1.3x")

    ds = CombinedDataset(root_dir=ROOT, augment=False, p_sbi=0.5)
    r, f, rl, fl = ds[0]
    print(f"real shape : {r.shape}")
    print(f"fake shape : {f.shape}")
    print(f"labels     : {rl.item()}, {fl.item()}")
    print("✅ dataset_combined.py OK")