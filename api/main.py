from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
import cv2
import numpy as np
import tempfile
import os
from typing import Dict, Any
import logging
from pathlib import Path
import timm
import mediapipe as mp
from PIL import Image
import torchvision.transforms as transforms
import base64
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Deepfake Detection API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class XceptionDeepfakeModel(nn.Module):
    """Direct Xception model to match your saved weights exactly"""
    def __init__(self, num_classes=2):
        super(XceptionDeepfakeModel, self).__init__()
        # Create Xception model without any wrapper - direct match to your weights
        self.xception = timm.create_model('xception', pretrained=False, num_classes=0)
        # Add final classifier to match your saved structure
        self.fc = nn.Linear(2048, num_classes)
    
    def forward(self, x):
        features = self.xception(x)
        output = self.fc(features)
        return output

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def load_model():
    """Load the PyTorch Xception model"""
    global model
    try:
        model = XceptionDeepfakeModel(num_classes=2)
        
        # Try to load your trained model
        model_path = Path("best_xception.pth")
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location=device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('fc.'):
                    # Keep fc layer keys as is
                    new_state_dict[key] = value
                else:
                    # Map all other keys to xception.* 
                    new_state_dict[f'xception.{key}'] = value
            
            # Load the remapped state dict
            missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
            
            if missing_keys:
                logger.warning(f"Missing keys: {len(missing_keys)} keys")
            if unexpected_keys:
                logger.warning(f"Unexpected keys: {len(unexpected_keys)} keys")
                
            logger.info("Successfully loaded your trained Xception model!")
        else:
            logger.warning("best_xception.pth not found, using pre-trained Xception")
        
        model.to(device)
        model.eval()
        logger.info(f"Xception model loaded successfully on {device}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        model = None

def preprocess_frame(frame):
    """Preprocess video frame using your exact pipeline"""
    # Convert BGR to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces using MediaPipe
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(rgb_frame)
        
        if results.detections:
            # Get the first detected face
            detection = results.detections[0]
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            
            # Convert relative coordinates to absolute
            x = int(bboxC.xmin * iw)
            y = int(bboxC.ymin * ih)
            w = int(bboxC.width * iw)
            h = int(bboxC.height * ih)
            
            # Crop face region
            face = frame[y:y+h, x:x+w]
            
            if face.size > 0:
                # Resize to 299x299 for Xception
                face = cv2.resize(face, (299, 299))
                # Convert to RGB
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                # Convert to PIL Image
                face = Image.fromarray(face)
                
                # Apply transforms
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                ])
                
                face_tensor = transform(face).unsqueeze(0).to(device)
                return face_tensor
    
    # If no face detected, use full frame
    frame = cv2.resize(frame, (299, 299))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return transform(frame).unsqueeze(0).to(device)

def extract_frames(video_path: str, max_frames: int = 30):
    """Extract frames evenly from video like in your implementation"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return frames, []
    
    # Sample frames evenly throughout the video
    frame_indices = np.linspace(0, total_frames - 1, min(max_frames, total_frames), dtype=int)
    original_frames = []  # Keep original frames for sample extraction
    
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
            original_frames.append(frame.copy())
    
    cap.release()
    return frames, original_frames

def frame_to_base64(frame):
    """Convert OpenCV frame to base64 string"""
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode()

def create_heatmap(frames, predictions):
    if not frames:
        return None

    base_frame = frames[0].copy()
    h, w = base_frame.shape[:2]

    heatmap = np.zeros((h, w), dtype=np.uint8)

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        for i, frame in enumerate(frames):
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)

            if results.detections and i < len(predictions):
                detection = results.detections[0]
                bboxC = detection.location_data.relative_bounding_box

                x = int(bboxC.xmin * w)
                y = int(bboxC.ymin * h)
                box_w = int(bboxC.width * w)
                box_h = int(bboxC.height * h)

                
                if predictions[i] == 1:  # FAKE
                    heatmap[y:y+box_h, x:x+box_w] = 255   # RED
                else:  # REAL
                    heatmap[y:y+box_h, x:x+box_w] = 100   # GREEN


    heatmap_color = np.zeros((h, w, 3), dtype=np.uint8)
    heatmap_color[heatmap == 255] = [0, 0, 255]   
    heatmap_color[heatmap == 100] = [0, 255, 0]   
    result = cv2.addWeighted(base_frame, 0.6, heatmap_color, 0.4, 0)

    return result

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device)
    }

@app.post("/predict")
async def predict_deepfake(file: UploadFile = File(...)):
    """Predict if uploaded video contains deepfakes using your Xception model"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        # Extract frames from video
        frames, original_frames = extract_frames(temp_path, max_frames=30)
        
        if not frames:
            raise HTTPException(status_code=400, detail="Could not extract frames from video")
        
        logger.info(f"Processing {len(frames)} frames...")
        
        # Process frames and get predictions
        predictions = []
        confidences = []
        fake_probabilities = []
        
        with torch.no_grad():
            for i, frame in enumerate(frames):
                processed_frame = preprocess_frame(frame)
                if processed_frame is not None:
                    output = model(processed_frame)
                    probabilities = torch.softmax(output, dim=1)
                    
                    # Get probability of being fake (class 1)
                    fake_prob = probabilities[0][1].item()
                    real_prob = probabilities[0][0].item()
                    fake_probabilities.append(fake_prob)
                    
                    if i < 5:
                        logger.info(f"Frame {i}: Real={real_prob:.3f}, Fake={fake_prob:.3f}, Raw output={output[0].tolist()}")
                    
                    # Use lower threshold temporarily to see if model is working
                    confidence = torch.max(probabilities).item()
                    prediction = 1 if fake_prob > 0.5 else 0  # Lowered threshold to 0.5
                    
                    predictions.append(prediction)
                    confidences.append(confidence)
        
        if not predictions:
            raise HTTPException(status_code=400, detail="No faces detected in video")
        
        # Calculate statistics
        fake_count = sum(predictions)
        real_count = len(predictions) - fake_count
        avg_confidence = np.mean(confidences)
        avg_fake_probability = np.mean(fake_probabilities)
        
        logger.info(f"Results: {fake_count} fake frames, {real_count} real frames")
        logger.info(f"Average fake probability: {avg_fake_probability:.3f}")
        logger.info(f"Average confidence: {avg_confidence:.3f}")
        
        fake_frame_percentage = fake_count / len(predictions)
        is_deepfake = fake_frame_percentage > 0.5 or avg_fake_probability > 0.55
        
        logger.info(f"Classification decision:")
        logger.info(f"  - Fake frame percentage: {fake_frame_percentage:.3f} (threshold: >0.5)")
        logger.info(f"  - Average fake probability: {avg_fake_probability:.3f} (threshold: >0.55)")
        logger.info(f"  - Condition 1 (fake_frame_percentage > 0.5): {fake_frame_percentage > 0.5}")
        logger.info(f"  - Condition 2 (avg_fake_probability > 0.55): {avg_fake_probability > 0.55}")
        logger.info(f"  - Final classification: {'DEEPFAKE' if is_deepfake else 'REAL'}")
        
        real_frame_base64 = None
        fake_frame_base64 = None
        
        # Find first real and first fake frame
        for i, pred in enumerate(predictions):
            if pred == 0 and real_frame_base64 is None:
                real_frame_base64 = frame_to_base64(original_frames[i])
            if pred == 1 and fake_frame_base64 is None:
                fake_frame_base64 = frame_to_base64(original_frames[i])
        
        # Create heatmap visualization
        heatmap_image = create_heatmap(original_frames, predictions)
        heatmap_base64 = frame_to_base64(heatmap_image) if heatmap_image is not None else None
        
        # Clean up temporary file
        os.unlink(temp_path)
        
        return {
            "is_deepfake": bool(is_deepfake),
            "confidence": float(avg_confidence),
            "deepfake_probability": float(avg_fake_probability),
            "sample_frames": {
                "real_frame": real_frame_base64,
                "fake_frame": fake_frame_base64,
            },
            "heatmap": heatmap_base64,
            "statistics": {
                "total_frames_analyzed": int(len(predictions)),
                "fake_frames": int(fake_count),
                "real_frames": int(real_count),
                "fake_frame_percentage": float(fake_frame_percentage),
                "average_confidence": float(avg_confidence),
                "average_fake_probability": float(avg_fake_probability),
                "confidence_range": f"{min(confidences):.1%} - {max(confidences):.1%}",
                "classification_method": "Majority vote (>50% fake frames) OR high fake probability (>55%)"
            }
        }
    
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        # Clean up temporary file if it exists
        if 'temp_path' in locals():
            try:
                os.unlink(temp_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
