# Deepfake Detection Web App

A comprehensive deepfake detection system with React frontend and ML backend.

## Features

- **Real-time Video Upload**: Drag & drop or click to upload video files
- **ML-Powered Detection**: Uses Xception model trained on deepfake datasets
- **Frame-by-Frame Analysis**: Processes multiple frames for accurate detection
- **Detailed Results**: Shows confidence scores, statistics, and frame-level predictions
- **Dark Theme UI**: Modern, responsive interface

## Current Status

### Preview Version (v0.app)
- ✅ Full UI functionality with video upload
- ✅ Realistic mock detection results that simulate your Xception model
- ✅ Complete results visualization and statistics
- ⚠️ Uses simulated ML inference (no actual PyTorch model)

### Production Version (Local Deployment)
- ✅ Real Xception model integration (`scripts/deepfake_model.py`)
- ✅ Complete video processing pipeline with MediaPipe + MTCNN
- ✅ PyTorch inference with your trained model
- 📋 Requires: Python environment with PyTorch, MediaPipe, MTCNN

## Local Deployment with Real Model

1. **Install Dependencies**:
   \`\`\`bash
   pip install torch torchvision timm opencv-python mediapipe mtcnn pillow numpy
   \`\`\`

2. **Add Your Model**:
   - Place your trained `best_xception.pth` file in the `scripts/` directory
   - Update the model path in `scripts/deepfake_model.py` if needed

3. **Run the Application**:
   \`\`\`bash
   npm install
   npm run dev
   \`\`\`

## Model Architecture

- **Base Model**: Xception (pretrained on ImageNet)
- **Input Size**: 299x299 pixels
- **Output**: Binary classification (Real: 0, Fake: 1)
- **Face Detection**: MediaPipe + MTCNN pipeline
- **Frame Sampling**: 12 evenly distributed frames per video

## API Endpoints

- `POST /api/predict` - Upload video for deepfake detection
  - Input: FormData with video file
  - Output: Detection results with confidence scores and frame analysis
