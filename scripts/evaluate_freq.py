import os, sys, torch, numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.metrics import roc_auc_score
from collections import defaultdict
from tqdm import tqdm
from facenet_pytorch import MTCNN

sys.path.insert(0, os.path.dirname(__file__))
from model_freq import FreqDetector

BASE       = "/media/DiscoLocal/MUDLAVSP/e542476/Task 2 DeepFakes Detection-20260226/Task 2 DeepFakes Detection-20260226"
TS_PATH    = os.path.join(BASE, "models/yermandy/model.torchscript")
MODEL_PATH = os.path.join(BASE, "models/best_freq.pth")
EVAL_DIR   = os.path.join(BASE, "data/Task_2_3/Task_2_3/evaluation")
CROP_SCALE = 1.3
TEMP       = 5.0
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                         std=[0.26862954, 0.26130258, 0.27577711]),
])

def crop_face(image, mtcnn, scale=1.3):
    boxes, _ = mtcnn.detect(image)
    if boxes is not None:
        box = boxes[0]
        cx=(box[0]+box[2])/2; cy=(box[1]+box[3])/2
        w=(box[2]-box[0])*scale; h=(box[3]-box[1])*scale
        return image.crop((cx-w/2, cy-h/2, cx+w/2, cy+h/2))
    return image

def get_frames(folder):
    frames = []
    for root, dirs, files in os.walk(folder):
        for f in sorted(files):
            if f.lower().endswith(('.jpg','.png','.jpeg')):
                frames.append(os.path.join(root, f))
    return frames

def video_agg(probs, temp=5.0):
    probs = np.array(probs)
    w = np.exp(temp * np.abs(probs - 0.5) * 2.0)
    w /= w.sum()
    return float(np.sum(w * probs))

def evaluate():
    print("=" * 60)
    print("  Evaluación FreqDetector (CLIP + DWT) en Celeb-DF")
    print(f"  Dispositivo: {DEVICE} | Crop: {CROP_SCALE}x")
    print("=" * 60)

    mtcnn = MTCNN(keep_all=False, post_process=False, device=DEVICE)
    model = FreqDetector(TS_PATH).to(DEVICE)

    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    model.freq_branch.load_state_dict(ckpt["freq_branch"])
    model.attention_did.load_state_dict(ckpt["attention_did"])
    model.fc.load_state_dict(ckpt["fc"])
    print(f"  Checkpoint época {ckpt.get('epoch','?')} | Val AUC: {ckpt.get('val_auc',0):.4f}")
    model.eval()

    def infer(frames, desc):
        probs = []
        with torch.no_grad():
            for path in tqdm(frames, desc=desc):
                img  = Image.open(path).convert("RGB")
                face = crop_face(img, mtcnn, CROP_SCALE)
                x    = transform(face).unsqueeze(0).to(DEVICE)
                with torch.amp.autocast("cuda"):
                    logit = model(x).squeeze()
                probs.append(torch.sigmoid(logit).item())
        return probs

    real_frames = get_frames(os.path.join(EVAL_DIR, "real"))
    fake_frames = get_frames(os.path.join(EVAL_DIR, "fake"))
    real_probs  = infer(real_frames, "real")
    fake_probs  = infer(fake_frames, "fake")

    frame_auc = roc_auc_score([1.0]*len(real_probs)+[0.0]*len(fake_probs),
                               real_probs+fake_probs)

    def group(frames, probs):
        d = defaultdict(list)
        for p, prob in zip(frames, probs):
            d[os.path.basename(os.path.dirname(p))].append(prob)
        return d

    vid_probs, vid_labels = [], []
    for v, p in group(real_frames, real_probs).items():
        vid_probs.append(video_agg(p, TEMP)); vid_labels.append(1.0)
    for v, p in group(fake_frames, fake_probs).items():
        vid_probs.append(video_agg(p, TEMP)); vid_labels.append(0.0)

    video_auc = roc_auc_score(vid_labels, vid_probs)

    print(f"\n  Frame-level AUC  : {frame_auc:.4f} ({frame_auc*100:.2f}%)")
    print(f"  Score real       : {np.mean(real_probs):.3f} ± {np.std(real_probs):.3f}")
    print(f"  Score fake       : {np.mean(fake_probs):.3f} ± {np.std(fake_probs):.3f}")
    print(f"  Video-level AUC  : {video_auc:.4f} ({video_auc*100:.2f}%) ← MÉTRICA PRINCIPAL")
    print(f"  Vídeos           : {len(vid_labels)}")

if __name__ == "__main__":
    evaluate()