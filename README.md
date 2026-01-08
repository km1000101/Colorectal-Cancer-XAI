# Colorectal Cancer Histology Classification System

A full-stack web application for colorectal cancer histology image classification using ensemble deep learning models with explainable AI (XAI) visualizations.

## Overview

This system combines four pre-trained PyTorch CNN architectures (ResNet50, MobileNetV2, EfficientNet-B3, DenseNet121) trained on the Kather colorectal histology dataset to classify histology patches into 9 classes. The application provides interactive visual explanations using Grad-CAM, LIME, and SHAP (GradientShap) techniques.

**Classification Classes:**
1. `01_TUMOR` - Tumor tissue
2. `02_STROMA` - Stroma tissue
3. `03_COMPLEX` - Complex tissue
4. `04_LYMPHO` - Lymphoid tissue
5. `05_DEBRIS` - Debris
6. `06_MUCOSA` - Mucosa tissue
7. `07_ADIPOSE` - Adipose tissue
8. `08_EMPTY` - Empty background
9. `UNKNOWN` - Unknown/uncertain classification

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Client Layer                            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         React Frontend (TypeScript + Vite)          │   │
│  │         Port: 5173                                   │   │
│  │  - Image Upload & Preview                           │   │
│  │  - Model Selection                                  │   │
│  │  - XAI Visualization (Grad-CAM, LIME, SHAP)        │   │
│  │  - Prediction Results Display                      │   │
│  │  - History Management                               │   │
│  │  - Report Generation                                │   │
│  └───────────────────┬─────────────────────────────────┘   │
└──────────────────────┼──────────────────────────────────────┘
                       │ HTTP/REST API
                       │ /api/*
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         FastAPI Backend (Python 3.11)                │   │
│  │         Port: 8000                                    │   │
│  │  - CORS Middleware                                    │   │
│  │  - Request Validation (Pydantic)                    │   │
│  │  - Image Preprocessing                               │   │
│  └───────────────────┬─────────────────────────────────┘   │
└──────────────────────┼──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
┌─────────────┐ ┌──────────────┐ ┌──────────────┐
│  Prediction │ │  XAI Module  │ │ Model Loader │
│   Router    │ │   Router     │ │   Manager    │
└──────┬──────┘ └──────┬───────┘ └──────┬───────┘
       │               │                 │
       └───────────────┼─────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                      Model Layer                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Ensemble Prediction Engine               │   │
│  │  - Single Model Inference                            │   │
│  │  - Ensemble Averaging (4 models)                    │   │
│  └───────────────────┬──────────────────────────────────┘   │
│                      │                                       │
│  ┌───────────────────┴──────────────────────────────────┐  │
│  │         PyTorch CNN Models (Pre-trained)              │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ │  │
│  │  │ResNet50  │ │MobileNet │ │Efficient │ │DenseNet  │ │  │
│  │  │          │ │   V2     │ │  Net-B3  │ │  121     │ │  │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    XAI Explanation Layer                     │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐        │
│  │   Grad-CAM   │ │     LIME     │ │     SHAP     │        │
│  │  (Gradient   │ │  (Local      │ │  (Gradient   │        │
│  │  Activation) │ │  Interpret.) │ │  Shapley)    │        │
│  └──────────────┘ └──────────────┘ └──────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

### Component Architecture

#### Backend Components

**1. API Layer (`backend/main.py`)**
- FastAPI application entry point
- CORS middleware configuration
- Route registration and health check endpoint
- API version: 0.2.0

**2. Router Modules (`backend/routers/`)**

   **`predict.py`** - Prediction Endpoint
   - `POST /api/predict` - Image classification without XAI
   - Returns: predicted class, confidence, probabilities, per-model scores

   **`xai.py`** - Explainable AI Endpoint
   - `POST /api/explain` - Classification with XAI visualizations
   - Parameters:
     - `file`: Image file (multipart/form-data)
     - `model_name`: Model selection (default: "ensemble")
     - `explanation_types`: List of XAI methods (gradcam, lime, shap)
   - Returns: Prediction results + base64-encoded explanation images

**3. Model Management (`backend/models/`)**

   **`loader.py`** - Model Loading & Inference
   - Lazy loading and caching of PyTorch models
   - Image preprocessing (resize to 224×224, normalization)
   - Single model and ensemble prediction functions
   - Model checkpoint management

   **`classifiers.py`** - Model Architecture Definitions
   - PyTorch model class definitions for each CNN architecture
   - Custom classifier heads for 9-class classification

   **`ensemble.py`** - Ensemble Prediction Wrapper
   - Aggregates predictions from all 4 models
   - Averages probabilities for ensemble prediction
   - Returns per-model breakdowns

**4. XAI Module (`backend/xai/`)**

   **`gradcam.py`** - Gradient-weighted Class Activation Mapping
   - Generates heatmaps showing important regions for classification
   - Uses gradient information from the final convolutional layer

   **`lime_explainer.py`** - Local Interpretable Model-agnostic Explanations
   - Segments image into superpixels
   - Perturbs segments to understand local importance
   - Generates overlay visualization

   **`gradient_shap_explainer.py`** - SHAP Explanations
   - Uses GradientShap from Captum library
   - Computes Shapley values for feature importance
   - Generates attribution heatmaps

**5. Data Schemas (`backend/schemas.py`)**
- Pydantic models for request/response validation
- Type-safe API contracts

#### Frontend Components

**1. Application Structure (`frontend/src/`)**

   **`App.tsx`** / **`pages/Home.tsx`** - Main Application
   - Image upload interface with drag-and-drop support
   - Model selection dropdown
   - XAI method selection checkboxes
   - Results display orchestration

   **`Layout.tsx`** - Application Layout
   - Navigation bar with smooth scrolling
   - Route management (React Router)
   - Page shell wrapper

**2. UI Components (`frontend/src/components/`)**

   **`PredictionCard.tsx`**
   - Displays predicted class and confidence score
   - Probability distribution chart for all 9 classes
   - Per-model probability comparison table
   - Visual probability bars

   **`ExplanationTabs.tsx`**
   - Tabbed interface for XAI visualizations
   - Grad-CAM overlay display
   - LIME segmentation overlay
   - SHAP heatmap visualization
   - Image comparison views

   **`History.tsx`**
   - Displays analysis history from localStorage
   - Previous predictions and explanations
   - Image thumbnails and metadata

   **`PatientInfoModal.tsx`**
   - Patient information input form
   - Metadata collection for reports

   **`DownloadReportButton.tsx`**
   - Generates PDF reports using jsPDF
   - Includes predictions, explanations, and patient info

**3. Services (`frontend/src/services/`)**

   **`historyService.ts`**
   - LocalStorage management for analysis history
   - Save/load prediction results

   **`reportService.ts`**
   - PDF report generation logic
   - Combines images, predictions, and metadata

**4. API Client (`frontend/src/api/client.ts`)**
- TypeScript API client functions
- Type-safe request/response interfaces
- FormData handling for file uploads
- Environment-based API URL configuration

### Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interaction Flow                      │
└─────────────────────────────────────────────────────────────┘

1. User Upload
   │
   ├─> Frontend: File selection/drag-drop
   ├─> Preview generation (URL.createObjectURL)
   └─> Base64 conversion for report generation

2. Model & Explanation Selection
   │
   ├─> Model: Ensemble or individual (ResNet50, MobileNetV2, etc.)
   └─> Explanations: Grad-CAM, LIME, SHAP (multi-select)

3. API Request
   │
   ├─> POST /api/explain
   ├─> FormData: file + model_name + explanation_types[]
   └─> Frontend: Loading state management

4. Backend Processing
   │
   ├─> Image Loading (PIL Image.open)
   ├─> Image Preprocessing (resize, normalize)
   │
   ├─> Model Inference
   │   ├─> Load models (cached after first load)
   │   ├─> Forward pass through selected model(s)
   │   └─> Probability computation
   │
   ├─> XAI Generation (parallel where possible)
   │   ├─> Grad-CAM: Gradient computation + heatmap
   │   ├─> LIME: Superpixel segmentation + perturbation
   │   └─> SHAP: GradientShap attribution
   │
   └─> Response Encoding
       ├─> Base64 encoding of explanation images
       └─> JSON response with prediction + explanations

5. Frontend Display
   │
   ├─> PredictionCard: Class probabilities, confidence
   ├─> ExplanationTabs: Visual overlays
   ├─> History: Save to localStorage
   └─> Report: PDF generation option
```

### Technology Stack

#### Backend
- **Framework**: FastAPI 0.104+
- **Python**: 3.11
- **Deep Learning**: PyTorch, torchvision
- **Image Processing**: Pillow (PIL), OpenCV, scikit-image
- **XAI Libraries**: 
  - `grad-cam` - Grad-CAM implementation
  - `lime` - LIME explanations
  - `captum` - SHAP (GradientShap)
- **API**: uvicorn (ASGI server)
- **Validation**: Pydantic

#### Frontend
- **Framework**: React 18.2+
- **Language**: TypeScript 5.6+
- **Build Tool**: Vite 5.0+
- **Routing**: React Router DOM 6.30+
- **PDF Generation**: jsPDF 2.5+
- **Styling**: CSS (custom stylesheet)

#### Deployment
- **Containerization**: Docker, Docker Compose
- **Backend Image**: python:3.11-slim
- **Frontend Dev**: node:20 (Vite dev server)
- **Production**: Frontend build served via static files

---

## Project Structure

```
Major Project Colorectal/
├── backend/                    # FastAPI backend application
│   ├── main.py                # FastAPI app entry point
│   ├── schemas.py             # Pydantic models
│   ├── requirements.txt       # Python dependencies
│   ├── .env                   # Environment variables
│   ├── models/                # Model management
│   │   ├── loader.py          # Model loading & inference
│   │   ├── classifiers.py    # CNN architecture definitions
│   │   └── ensemble.py        # Ensemble prediction wrapper
│   ├── routers/               # API route handlers
│   │   ├── predict.py         # Prediction endpoint
│   │   └── xai.py             # XAI explanation endpoint
│   └── xai/                   # Explainable AI modules
│       ├── gradcam.py         # Grad-CAM implementation
│       ├── lime_explainer.py  # LIME explanations
│       └── gradient_shap_explainer.py  # SHAP explanations
│
├── frontend/                   # React frontend application
│   ├── src/
│   │   ├── App.tsx            # Main app component (legacy)
│   │   ├── main.tsx           # React entry point
│   │   ├── Layout.tsx         # Application layout
│   │   ├── api/
│   │   │   └── client.ts      # API client functions
│   │   ├── components/        # React components
│   │   │   ├── PredictionCard.tsx
│   │   │   ├── ExplanationTabs.tsx
│   │   │   ├── History.tsx
│   │   │   ├── PatientInfoModal.tsx
│   │   │   └── DownloadReportButton.tsx
│   │   ├── pages/             # Page components
│   │   │   ├── Home.tsx       # Main analysis page
│   │   │   ├── About.tsx      # About page
│   │   │   └── Courses.tsx    # Courses page
│   │   ├── services/          # Business logic services
│   │   │   ├── historyService.ts
│   │   │   └── reportService.ts
│   │   └── styles.css         # Global styles
│   ├── package.json           # Node.js dependencies
│   ├── vite.config.ts         # Vite configuration
│   └── tsconfig.json          # TypeScript configuration
│
├── models(ResNet50)/          # Pre-trained model checkpoints
│   └── best_colorectal_model.pth
├── models(MobileNetV2)/
│   └── best_colorectal_model.pth
├── models(EfficientNetB3)/
│   └── best_colorectal_model.pth
├── models(DenseNet121)/
│   └── best_colorectal_model.pth
│
├── docker-compose.yml         # Docker Compose configuration
├── Dockerfile                 # Backend Docker image
├── README.md                  # This file
│
└── *.ipynb                    # Jupyter notebooks (training/evaluation)
    ├── colorectal-cancer-densenet121.ipynb
    ├── colorectal-cancer-efficientnetb3.ipynb
    ├── colorectal-cancer-ensembling-and-xai.ipynb
    ├── colorectal-cancer-mobilenetv2.ipynb
    └── colorectal-cancer-resnet50.ipynb
```

---

## Installation & Setup

### Prerequisites

- **Python**: 3.11 or higher
- **Node.js**: 20.x or higher
- **Docker** (optional): For containerized deployment
- **CUDA** (optional): For GPU acceleration (requires CUDA-compatible PyTorch)

### Local Development Setup

#### 1. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables (if needed)
# Create .env file with any required configuration

# Run the development server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The backend API will be available at `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs` (Swagger UI)
- Alternative docs: `http://localhost:8000/redoc`

#### 2. Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Run the development server
npm run dev
```

The frontend will be available at `http://localhost:5173`

**Note**: The frontend proxies `/api/*` requests to `http://localhost:8000` by default. Configure `VITE_API_BASE_URL` in `.env` if needed.

#### 3. Model Checkpoints

Ensure the pre-trained model checkpoints are present in their respective directories:
- `models(ResNet50)/best_colorectal_model.pth`
- `models(MobileNetV2)/best_colorectal_model.pth`
- `models(EfficientNetB3)/best_colorectal_model.pth`
- `models(DenseNet121)/best_colorectal_model.pth`

---

## Docker Deployment

### Using Docker Compose (Recommended)

```bash
# Build and start all services
docker-compose up --build

# Run in detached mode
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

This will start:
- **Backend**: `http://localhost:8000`
- **Frontend**: `http://localhost:5173`

### Dockerfile Details

The `Dockerfile` builds a Python 3.11-slim image containing:
- All backend dependencies
- Backend application code
- Pre-trained model checkpoints

The `docker-compose.yml` orchestrates:
- Backend service (from Dockerfile)
- Frontend service (Node.js dev server)

---

## Usage Guide

### Web Application

1. **Upload Image**
   - Click the upload area or drag-and-drop a histology image
   - Supported formats: PNG, JPG, JPEG, TIFF
   - Recommended: 224×224 Kather colorectal histology tiles

2. **Select Model**
   - Choose **Ensemble** (default) for combined predictions from all 4 models
   - Or select a specific model: ResNet50, MobileNetV2, EfficientNetB3, or DenseNet121

3. **Choose Explanations**
   - Select one or more XAI methods:
     - **Grad-CAM**: Gradient-weighted activation maps
     - **LIME**: Local interpretable explanations
     - **SHAP**: Shapley value attributions

4. **Run Analysis**
   - Click "Run analysis" button
   - Wait for inference and explanation generation
   - Results appear in the prediction card and explanation tabs

5. **View Results**
   - **Prediction Card**: Shows predicted class, confidence, probability distribution, and per-model breakdown
   - **Explanation Tabs**: Visual overlays showing important regions for classification
   - **History**: Previous analyses are saved automatically

6. **Generate Report** (if available)
   - Fill in patient information (optional)
   - Download PDF report with predictions and explanations

### API Usage

#### Prediction Endpoint

```bash
curl -X POST "http://localhost:8000/api/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/image.jpg"
```

**Response:**
```json
{
  "predicted_class": "01_TUMOR",
  "confidence": 0.95,
  "class_probabilities": {
    "01_TUMOR": 0.95,
    "02_STROMA": 0.03,
    ...
  },
  "per_model_scores": [
    {
      "model_name": "ResNet50",
      "probabilities": {...}
    },
    ...
  ]
}
```

#### XAI Explanation Endpoint

```bash
curl -X POST "http://localhost:8000/api/explain?model_name=ensemble&explanation_types=gradcam&explanation_types=lime&explanation_types=shap" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/image.jpg"
```

**Response:**
```json
{
  "prediction": {
    "predicted_class": "01_TUMOR",
    "confidence": 0.95,
    ...
  },
  "gradcam": {
    "heatmap_base64": "iVBORw0KGgoAAAANS..."
  },
  "lime": {
    "overlay_base64": "iVBORw0KGgoAAAANS..."
  },
  "shap": {
    "heatmap_base64": "iVBORw0KGgoAAAANS..."
  }
}
```

---

## Development Notes

### Model Training

The models were trained using Jupyter notebooks:
- Individual model training notebooks for each architecture
- Ensemble and XAI evaluation notebook
- Training on Kather colorectal histology dataset (8 classes + UNKNOWN)

### Performance Considerations

- **Model Loading**: Models are loaded lazily and cached in memory after first use
- **Inference**: Single model inference is faster; ensemble requires 4 forward passes
- **XAI Generation**: Grad-CAM is fastest; LIME and SHAP are computationally intensive
- **Image Size**: Models expect 224×224 input; larger images are resized

### Future Enhancements

- GPU acceleration support
- Batch prediction endpoint
- Additional XAI methods (Integrated Gradients, Attention maps)
- Model versioning and A/B testing
- User authentication and multi-user support
- Database integration for history persistence
- Real-time explanation streaming

---

## License

[Add your license information here]

## Contributors

[Add contributor information here]

## Acknowledgments

- Kather et al. for the colorectal histology dataset
- PyTorch and torchvision teams
- FastAPI and React communities
- XAI library maintainers (grad-cam, LIME, Captum)

---

## Support

For issues, questions, or contributions, please [open an issue](link-to-repo/issues) or contact the development team.
