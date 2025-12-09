import torch
import torch.nn as nn
import timm
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
import os
import tempfile
from torchvision import transforms

class DeepfakeDetector:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize MediaPipe face detection
        self.mp_face = mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=0.5
        )
        
        # Load Xception model
        self.model = timm.create_model("xception", pretrained=True, num_classes=2)
        
        # If model path provided, load trained weights
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded trained model from {model_path}")
        else:
            print("Using pretrained Xception model (not fine-tuned for deepfakes)")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Define transforms (Xception expects 299x299)
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def detect_and_crop_face(self, bgr_frame, out_size=299):
        """Detect and crop face from frame using MediaPipe"""
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        results = self.mp_face.process(rgb)
        
        if not results.detections:
            return None
        
        # Get the first (most confident) detection
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        
        h, w = bgr_frame.shape[:2]
        x1 = max(int(bbox.xmin * w) - 10, 0)
        y1 = max(int(bbox.ymin * h) - 10, 0)
        x2 = min(int((bbox.xmin + bbox.width) * w) + 10, w)
        y2 = min(int((bbox.ymin + bbox.height) * h) + 10, h)
        
        face = bgr_frame[y1:y2, x1:x2]
        if face.size == 0:
            return None
        
        face = cv2.resize(face, (out_size, out_size))
        return face
    
    def get_even_indices(self, total_frames, K=12):
        """Get evenly spaced frame indices"""
        if total_frames == 0 or K == 0:
            return []
        if K >= total_frames:
            return list(range(total_frames))
        step = total_frames / K
        return [int(i * step) for i in range(K)]
    
    def extract_frames_from_video(self, video_path, max_frames=12):
        """Extract frames from video and detect faces"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            return []
        
        # Get evenly spaced frame indices
        frame_indices = self.get_even_indices(total_frames, max_frames)
        
        faces = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # Detect and crop face
            face = self.detect_and_crop_face(frame)
            if face is not None:
                faces.append(face)
        
        cap.release()
        return faces
    
    def predict_single_face(self, face_bgr):
        """Predict if a single face is deepfake"""
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb)
        
        # Apply transforms
        face_tensor = self.transform(face_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(face_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            # probabilities[0][0] = real, probabilities[0][1] = fake
            fake_prob = probabilities[0][1].item()
            real_prob = probabilities[0][0].item()
            
            return {
                'fake_probability': fake_prob,
                'real_probability': real_prob,
                'is_fake': fake_prob > 0.5,
                'confidence': max(fake_prob, real_prob)
            }
    
    def analyze_video(self, video_path):
        """Analyze entire video for deepfake detection"""
        print(f"Analyzing video: {video_path}")
        
        # Extract faces from video
        faces = self.extract_frames_from_video(video_path)
        
        if not faces:
            return {
                'error': 'No faces detected in video',
                'is_deepfake': False,
                'confidence': 0.0,
                'deepfake_probability': 0.0,
                'statistics': {
                    'total_frames_analyzed': 0,
                    'fake_frames': 0,
                    'real_frames': 0,
                    'average_confidence': 0.0,
                    'min_confidence': 0.0,
                    'max_confidence': 0.0
                }
            }
        
        # Analyze each face
        results = []
        for face in faces:
            result = self.predict_single_face(face)
            results.append(result)
        
        # Aggregate results
        fake_count = sum(1 for r in results if r['is_fake'])
        real_count = len(results) - fake_count
        
        avg_fake_prob = np.mean([r['fake_probability'] for r in results])
        confidences = [r['confidence'] for r in results]
        
        # Overall decision: majority vote with confidence weighting
        overall_fake_prob = avg_fake_prob
        is_deepfake = overall_fake_prob > 0.5
        overall_confidence = np.mean(confidences)
        
        return {
            'is_deepfake': is_deepfake,
            'confidence': overall_confidence,
            'deepfake_probability': overall_fake_prob,
            'statistics': {
                'total_frames_analyzed': len(results),
                'fake_frames': fake_count,
                'real_frames': real_count,
                'average_confidence': np.mean(confidences),
                'min_confidence': np.min(confidences),
                'max_confidence': np.max(confidences)
            }
        }

def main():
    
    import sys
    if len(sys.argv) != 2:
        print("Usage: python deepfake_model.py <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    # Initialize detector (you can provide path to your trained model)
    # detector = DeepfakeDetector("/path/to/your/best_xception.pth")
    detector = DeepfakeDetector()  # Using pretrained for now
    
    # Analyze video
    results = detector.analyze_video(video_path)
    
    # Print results as JSON
    import json
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
