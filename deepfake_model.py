from transformers import pipeline
import cv2

# MODÈLE PUBLIC QUI MARCHE À 100%
detector = pipeline(
    "image-classification",
    model="umm-maybe/deepfake-detection-v2",  # Version publique
    device=-1  # CPU only
)

def extract_score(video_path):
    cap = cv2.VideoCapture(video_path)
    scores = []
    frame_count = 0
    while cap.isOpened() and frame_count < 30:
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = detector(rgb_frame)
        # Cherche "FAKE" ou "REAL"
        fake_score = next((r['score'] for r in result if r['label'].lower() == 'fake'), 0.0)
        scores.append(fake_score)
    cap.release()
    return sum(scores) / len(scores) if scores else 0.5
