import os
import cv2
from tqdm import tqdm

def extract_frames(video_path, output_folder, num_frames=15):
    """Abre el vídeo, calcula los saltos y extrae N fotogramas equitativos."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        return
        
    # Calculamos cada cuántos frames debemos capturar uno
    step = max(total_frames // num_frames, 1)
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    count = 0
    extracted = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Si toca capturar este frame
        if count % step == 0 and extracted < num_frames:
            frame_path = os.path.join(output_folder, f"{video_name}_frame{extracted:03d}.jpg")
            cv2.imwrite(frame_path, frame)
            extracted += 1
            
        count += 1
        if extracted >= num_frames:
            break
            
    cap.release()

def process_folder(input_dir, output_dir, num_frames=15):
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(input_dir):
        print(f"⚠️ No encuentro la carpeta: {input_dir}")
        return

    videos = [f for f in os.listdir(input_dir) if f.endswith('.mp4')]
    print(f"Encontrados {len(videos)} vídeos. Extrayendo {num_frames} frames de cada uno...")
    
    for vid in tqdm(videos, desc="Progreso"):
        extract_frames(os.path.join(input_dir, vid), output_dir, num_frames)

if __name__ == '__main__':
    # 1. Rutas de origen (La ruta real donde se descargaron los vídeos)
    base_dataset = r'/media/DiscoLocal/MUDLAVSP/e542476/Task 2 DeepFakes Detection-20260226/Task 2 DeepFakes Detection-20260226/FaceForensics_Dataset'
    real_videos_dir = os.path.join(base_dataset, 'original_sequences', 'youtube', 'c23', 'videos')
    fake_videos_dir = os.path.join(base_dataset, 'manipulated_sequences', 'Deepfakes', 'c23', 'videos')
    
    # 2. Rutas de destino (Creamos la carpeta de frames justo al lado del dataset)
    output_base = r'/media/DiscoLocal/MUDLAVSP/e542476/Task 2 DeepFakes Detection-20260226/Task 2 DeepFakes Detection-20260226/FaceForensics_Frames'
    real_out_dir = os.path.join(output_base, 'Real')
    fake_out_dir = os.path.join(output_base, 'Fake')
    
    print("--- PROCESANDO VÍDEOS REALES ---")
    process_folder(real_videos_dir, real_out_dir, num_frames=15)
    
    print("\n--- PROCESANDO VÍDEOS FAKES ---")
    process_folder(fake_videos_dir, fake_out_dir, num_frames=15)
    
    print("\n¡Extracción completada con éxito! Imágenes listas para el entrenamiento.")