import sys, os, torch, numpy as np
sys.path.insert(0, '/tmp/selimsef')
sys.path.insert(0, os.path.dirname(__file__))
from training.zoo.classifiers import DeepFakeClassifier
from model_combined import CombinedDetector
from sklearn.metrics import roc_auc_score
from collections import defaultdict
from PIL import Image
from torchvision import transforms
from facenet_pytorch import MTCNN
from tqdm import tqdm
import glob

BASE        = "/media/DiscoLocal/MUDLAVSP/e542476/Task 2 DeepFakes Detection-20260226/Task 2 DeepFakes Detection-20260226"
WEIGHTS_DIR = "/tmp/selimsef/weights"
EVAL        = os.path.join(BASE, "data/Task_2_3/Task_2_3/evaluation")
CLIP_CKPT   = os.path.join(BASE, "models/combined_epoch4.pth")
TS_PATH     = os.path.join(BASE, "models/yermandy/model.torchscript")
W_EFF       = 0.6  # peso selimsef
W_CLIP      = 0.4  # peso CombinedDetector
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

norm_clip = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
    transforms.Normalize([0.48145466,0.4578275,0.40821073],[0.26862954,0.26130258,0.27577711])])
norm_eff  = transforms.Compose([transforms.Resize((380,380)), transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

def load_models():
    mtcnn = MTCNN(keep_all=False, post_process=False, device=DEVICE)
    eff_models = []
    for w in sorted(glob.glob(os.path.join(WEIGHTS_DIR, 'final_*'))):
        m = DeepFakeClassifier(encoder='tf_efficientnet_b7_ns').to(DEVICE)
        ckpt = torch.load(w, map_location=DEVICE)
        state = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        state = {k.replace('module.', ''): v for k, v in state.items()}
        m.load_state_dict(state); m.eval(); eff_models.append(m)
    print(f"  Selimsef: {len(eff_models)} modelos cargados")
    clip_model = CombinedDetector(TS_PATH).to(DEVICE)
    ckpt = torch.load(CLIP_CKPT, map_location=DEVICE)
    clip_model.attention_did.load_state_dict(ckpt['attention_did'])
    clip_model.fc.load_state_dict(ckpt['fc'])
    clip_model.eval()
    print(f"  CLIP CombinedDetector cargado (época {ckpt.get('epoch','?')})")
    return mtcnn, eff_models, clip_model

def get_frames(folder):
    frames = []
    for root, dirs, files in os.walk(folder):
        for f in sorted(files):
            if f.lower().endswith(('.jpg','.png','.jpeg')):
                frames.append(os.path.join(root, f))
    return frames

def infer(mtcnn, eff_models, clip_model, frames, desc):
    probs = []
    with torch.no_grad():
        for path in tqdm(frames, desc=desc):
            img = Image.open(path).convert('RGB')
            b,_ = mtcnn.detect(img)
            if b is not None:
                bx=b[0]; cx=(bx[0]+bx[2])/2; cy=(bx[1]+bx[3])/2
                w=(bx[2]-bx[0])*1.3; h=(bx[3]-bx[1])*1.3
                img=img.crop((cx-w/2,cy-h/2,cx+w/2,cy+h/2))
            x_e = norm_eff(img).unsqueeze(0).to(DEVICE)
            x_c = norm_clip(img).unsqueeze(0).to(DEVICE)
            eff_score  = np.mean([1.0-torch.sigmoid(m(x_e)).item() for m in eff_models])
            with torch.amp.autocast('cuda'):
                clip_score = torch.sigmoid(clip_model(x_c)).item()
            probs.append(W_EFF * eff_score + W_CLIP * clip_score)
    return probs

def video_agg(probs, temp=5.0):
    probs = np.array(probs)
    w = np.exp(temp * np.abs(probs - 0.5) * 2.0)
    w /= w.sum()
    return 0.6*float(np.sum(w*probs)) + 0.3*float(np.min(probs)) + 0.1*float(np.max(probs))

def group_by_video(frames, probs):
    d = defaultdict(list)
    for p, prob in zip(frames, probs):
        d[os.path.basename(os.path.dirname(p))].append(prob)
    return d

def evaluate():
    print("=" * 60)
    print("  Ensemble Final: Selimsef + CombinedDetector")
    print(f"  w_eff={W_EFF} w_clip={W_CLIP} | Dispositivo: {DEVICE}")
    print("=" * 60)

    mtcnn, eff_models, clip_model = load_models()

    real_frames = get_frames(os.path.join(EVAL, "real"))
    fake_frames = get_frames(os.path.join(EVAL, "fake"))
    real_probs  = infer(mtcnn, eff_models, clip_model, real_frames, "real")
    fake_probs  = infer(mtcnn, eff_models, clip_model, fake_frames, "fake")

    frame_auc = roc_auc_score([1.0]*len(real_probs)+[0.0]*len(fake_probs),
                               real_probs+fake_probs)

    vid_probs, vid_labels = [], []
    for v, p in group_by_video(real_frames, real_probs).items():
        vid_probs.append(video_agg(p)); vid_labels.append(1.0)
    for v, p in group_by_video(fake_frames, fake_probs).items():
        vid_probs.append(video_agg(p)); vid_labels.append(0.0)

    video_auc = roc_auc_score(vid_labels, vid_probs)

    print(f"\n  Frame-level AUC  : {frame_auc:.4f} ({frame_auc*100:.2f}%)")
    print(f"  Score real       : {np.mean(real_probs):.3f} ± {np.std(real_probs):.3f}")
    print(f"  Score fake       : {np.mean(fake_probs):.3f} ± {np.std(fake_probs):.3f}")
    print(f"  Video-level AUC  : {video_auc:.4f} ({video_auc*100:.2f}%) ← RESULTADO FINAL")
    print(f"  Vídeos           : {len(vid_labels)}")
    print("=" * 60)

if __name__ == "__main__":
    evaluate()
