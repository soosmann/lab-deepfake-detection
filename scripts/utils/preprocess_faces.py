import os
import cv2
from PIL import Image
from tqdm import tqdm
import torch
from facenet_pytorch import MTCNN

def preprocess_dataset(input_base, output_base):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Iniciando MTCNN en: {device}")
    
    # Creamos el detector una sola vez
    mtcnn = MTCNN(keep_all=False, post_process=False, device=device)
    
    for category in ['Real', 'Fake']:
        in_dir = os.path.join(input_base, category)
        out_dir = os.path.join(output_base, category)
        os.makedirs(out_dir, exist_ok=True)
        
        if not os.path.exists(in_dir):
            continue
            
        images = os.listdir(in_dir)
        print(f"\nProcesando {len(images)} imágenes en {category}...")
        
        for img_name in tqdm(images):
            in_path = os.path.join(in_dir, img_name)
            out_path = os.path.join(out_dir, img_name)
            
            # Si ya la hemos procesado antes, nos la saltamos (útil si se corta)
            if os.path.exists(out_path):
                continue
                
            try:
                image = Image.open(in_path).convert('RGB')
                boxes, _ = mtcnn.detect(image)
                
                if boxes is not None:
                    box = boxes[0]
                    # Bounding box 1.3x (El truco del Estado del Arte)
                    x_center = (box[0] + box[2]) / 2
                    y_center = (box[1] + box[3]) / 2
                    w = (box[2] - box[0]) * 1.3
                    h = (box[3] - box[1]) * 1.3
                    
                    new_box = (x_center - w/2, y_center - h/2, x_center + w/2, y_center + h/2)
                    face = image.crop(new_box).resize((224, 224))
                    face.save(out_path)
                else:
                    # Si no encuentra cara, guarda la imagen entera redimensionada
                    image.resize((224, 224)).save(out_path)
            except Exception as e:
                pass

if __name__ == '__main__':
    # De dónde cogemos los fotogramas enteros
    INPUT_DIR = r'/media/DiscoLocal/MUDLAVSP/e542476/Task 2 DeepFakes Detection-20260226/Task 2 DeepFakes Detection-20260226/FaceForensics_Frames'
    
    # Dónde guardamos las caras ya recortadas (Nueva carpeta)
    OUTPUT_DIR = r'/media/DiscoLocal/MUDLAVSP/e542476/Task 2 DeepFakes Detection-20260226/Task 2 DeepFakes Detection-20260226/FaceForensics_Faces_1.3x'
    
    preprocess_dataset(INPUT_DIR, OUTPUT_DIR)
    print("\n¡Todas las caras han sido recortadas y guardadas!")