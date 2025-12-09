# Python-Only Deepfake Detection Setup

This app now uses a pure Python FastAPI backend with your Xception deepfake detection model.

## Setup Instructions

### 1. Install Python Dependencies
\`\`\`bash
cd api
pip install -r requirements.txt
\`\`\`

### 2. Add Your Trained Model
- Copy your `best_xception.pth` model file to the `api/` directory
- The FastAPI backend will automatically load it on startup

### 3. Start the FastAPI Backend
\`\`\`bash
cd api
python main.py
\`\`\`
The API will run on `http://localhost:8000`

### 4. Start the Next.js Frontend
\`\`\`bash
npm install
npm run dev
\`\`\`
The frontend will run on `http://localhost:3000`

## Model Architecture

The backend now uses:
- **Xception backbone** with timm library
- **MediaPipe face detection** for preprocessing
- **Your exact preprocessing pipeline** (299x299 resize, face cropping, normalization)
- **Frame sampling** (30 frames evenly distributed)
- **Majority voting** for final prediction

## API Endpoints

- `GET /health` - Check if model is loaded
- `POST /predict` - Upload video for deepfake detection

## Response Format
\`\`\`json
{
  "is_deepfake": true,
  "confidence": 0.85,
  "deepfake_probability": 0.73,
  "statistics": {
    "total_frames_analyzed": 30,
    "fake_frames": 22,
    "real_frames": 8,
    "average_confidence": 0.85,
    "confidence_range": "67.2% - 94.1%"
  }
}
\`\`\`

## Benefits of Python-Only Approach

✅ **Direct PyTorch integration** - No subprocess calls  
✅ **Your exact model architecture** - Xception with custom classifier  
✅ **Real face detection** - MediaPipe preprocessing  
✅ **Better performance** - Native Python ML pipeline  
✅ **Easier debugging** - Single language stack for ML
