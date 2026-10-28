import os
import sys
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from collections import defaultdict
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from facenet_pytorch import MTCNN

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_gend import LNCLIPDetector

BASE       = "/media/DiscoLocal/MUDLAVSP/e542476/Task 2 DeepFakes Detection-20260226/Task 2 DeepFakes Detection-20260226"
EVAL_DIR   = os.path.join(BASE, "data/Task_2_3/Task_2_3/evaluation")
MODEL_PATH = os.path.join(BASE, "models/gend_epoch5.pth")
CROP_SCALE = 1.3
TEMPERATURE= 5.0
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLIP_MEAN = [0.48145466, 0.4578275,  0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
])


def crop_face(image, mtcnn, scale=1.3):
    boxes, _ = mtcnn.detect(image)
    if boxes is not None:
        box = boxes[0]
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        w  = (box[2] - box[0]) * scale
        h  = (box[3] - box[1]) * scale
        return image.crop((cx - w/2, cy - h/2, cx + w/2, cy + h/2))
    return image


def get_frames(folder):
    frames = []
    for root, dirs, files in os.walk(folder):
        for f in sorted(files):
            if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                frames.append(os.path.join(root, f))
    return frames


def video_agg_weighted(probs, temp=5.0):
    probs   = np.array(probs)
    confs   = np.abs(probs - 0.5) * 2.0
    weights = np.exp(temp * confs)
    weights /= weights.sum()
    return float(np.sum(weights * probs))


def evaluate():
    print("=" * 60)
    print("  Evaluación GenD en Celeb-DF (con MTCNN crop 1.3x)")
    print(f"  Dispositivo : {DEVICE}")
    print(f"  Modelo      : {os.path.basename(MODEL_PATH)}")
    print("=" * 60)

    mtcnn = MTCNN(keep_all=False, post_process=False, device=DEVICE)

    model = LNCLIPDetector().to(DEVICE)
    ckpt  = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Checkpoint época {ckpt.get('epoch','?')} | Val AUC train: {ckpt.get('val_auc', 0):.4f}")
    model.eval()

    real_frames = get_frames(os.path.join(EVAL_DIR, "real"))
    fake_frames = get_frames(os.path.join(EVAL_DIR, "fake"))

    def infer(frames, desc):
        probs = []
        with torch.no_grad():
            for path in tqdm(frames, desc=desc):
                img  = Image.open(path).convert("RGB")
                face = crop_face(img, mtcnn, CROP_SCALE)
                x    = transform(face).unsqueeze(0).to(DEVICE)
                with torch.amp.autocast("cuda"):
                    logits, _ = model(x)
                prob = torch.sigmoid(logits).item()
                probs.append(prob)
        return probs

    real_probs = infer(real_frames, "real")
    fake_probs = infer(fake_frames, "fake")

    all_probs  = real_probs + fake_probs
    all_labels = [1.0] * len(real_probs) + [0.0] * len(fake_probs)
    frame_auc  = roc_auc_score(all_labels, all_probs)

    # Video-level
    def group_by_video(frames, probs):
        by_vid = defaultdict(list)
        for path, prob in zip(frames, probs):
            vid = os.path.basename(os.path.dirname(path))
            by_vid[vid].append(prob)
        return by_vid

    real_by_vid = group_by_video(real_frames, real_probs)
    fake_by_vid = group_by_video(fake_frames, fake_probs)

    vid_probs, vid_labels = [], []
    for vid, probs in real_by_vid.items():
        vid_probs.append(video_agg_weighted(probs, TEMPERATURE))
        vid_labels.append(1.0)
    for vid, probs in fake_by_vid.items():
        vid_probs.append(video_agg_weighted(probs, TEMPERATURE))
        vid_labels.append(0.0)

    video_auc = roc_auc_score(vid_labels, vid_probs)

    print("\n" + "=" * 60)
    print("  RESULTADOS")
    print("=" * 60)
    print(f"  Frame-level AUC  : {frame_auc:.4f} ({frame_auc*100:.2f}%)")
    print(f"  Score medio real : {np.mean(real_probs):.3f} ± {np.std(real_probs):.3f}")
    print(f"  Score medio fake : {np.mean(fake_probs):.3f} ± {np.std(fake_probs):.3f}")
    print(f"  Video-level AUC  : {video_auc:.4f} ({video_auc*100:.2f}%) ← MÉTRICA PRINCIPAL")
    print(f"  Vídeos evaluados : {len(vid_labels)}")
    print("=" * 60)


if __name__ == "__main__":
    evaluate()