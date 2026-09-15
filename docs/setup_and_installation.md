# Setup & Installation

**[← Back to Project Report](./index.md)**

## Getting Started

### Prerequisites

Make sure the following are installed:

- Python 3.11
- pip
- Node.js 16+
- npm
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/YourUsername/Scene-Classification.git
cd Scene-Classification
```

> Replace `YourUsername` with the actual GitHub username/repository URL.

### 2. Create the Python Environment

```bash
python -m venv venv
```

#### Windows PowerShell

```powershell
.\venv\Scripts\Activate.ps1
```

#### macOS/Linux

```bash
source venv/bin/activate
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 3. Dataset Preparation (Required for Training)

If you intend to run the Jupyter notebooks to train the models from scratch, you will need the dataset. _(Note: If you only want to run the React/FastAPI web application using the pre-trained weights, you can skip this step)._

1. **Download the Data:** Download the pre-processed PyTorch-ready dataset from here: [Organized Dataset – Google Drive](https://drive.google.com/file/d/1Hh9pWRZChD8zzoSHDsptuxQzO1qCtkSA/view?usp=drive_link)
2. **Extract the Data:** Extract the downloaded `.zip` or `.tar` file.
3. **Placement:** Place the extracted folder into the root directory of the project, or update the `test_dir` and `train_dir` paths in the Jupyter notebooks to point to your local download location.

- **Raw Dataset Reference:** [Kaggle Places365 Dataset](https://www.kaggle.com/datasets/pankajkumar2002/places365)

### 4. Backend Setup

Navigate to the backend:

```bash
cd backend
```

Make sure the trained `.pth` model weights are present in the `backend` directory.

Start the FastAPI server:

```bash
uvicorn main:app --reload
```

The backend will be available at:

```text
http://127.0.0.1:8000
```

The models are initialized during application startup.

### 5. Frontend Setup

Open a separate terminal and navigate to the frontend:

```bash
cd frontend
```

Install the dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

The React application will normally be available at:

```text
http://localhost:5173
```

## Project Structure

```text
Scene_Classification/
│
├── backend/
│   ├── main.py
│   ├── best_scene_classifier.pth
│   └── mobilenet_scene_classifier-final.pth
│
├── frontend/
│   ├── src/
│   │   └── App.jsx
│   └── package.json
│
├── notebooks/
│   ├── MobileNet_model_training-final.ipynb
│   └── DINOV2_model_training.ipynb
│
├── results/
│   ├── mobile_net_main.png
│   └── dinov2_main.png
│
├── Scene_Classification_Progress_Report.pdf
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

### Directory Description

| Directory/File                             | Description                                                              |
| ------------------------------------------ | ------------------------------------------------------------------------ |
| `backend/`                                 | FastAPI backend and PyTorch inference pipeline                           |
| `frontend/`                                | React/Vite web application                                               |
| `notebooks/`                               | Dataset preparation, model training, evaluation, and profiling notebooks |
| `results/`                                 | Training accuracy and loss curves                                        |
| `*.pth`                                    | Trained model weights                                                    |
| `Scene_Classification_Progress_Report.pdf` | Detailed project progress report                                         |
| `requirements.txt`                         | Python dependencies                                                      |
| `.gitignore`                               | Files excluded from version control                                      |

## Author

**[Bhargav Das](https://github.com/Bhargav-Das-12)**

## License

MIT License.
