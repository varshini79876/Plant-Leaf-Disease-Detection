# 🌿 LeafScan — Plant Leaf Disease Detection System

A full-stack Django web application that uses a fine-tuned **ResNet50 CNN** to detect plant leaf diseases across **38 classes** and **14 plant species**, powered by the [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) on Kaggle.

---

## 📋 Table of Contents

1. [Features](#features)
2. [Project Structure](#project-structure)
3. [Tech Stack](#tech-stack)
4. [Prerequisites](#prerequisites)
5. [Installation & Setup](#installation--setup)
6. [Training the Model](#training-the-model)
7. [Running the Application](#running-the-application)
8. [Dataset Details](#dataset-details)
9. [Supported Plants & Diseases](#supported-plants--diseases)
10. [How the System Works](#how-the-system-works)
11. [Demo Mode](#demo-mode)

---

## ✨ Features

- 📸 **Drag-and-drop image upload** (JPG, PNG, WebP — up to 10 MB)
- 🔬 **38-class disease detection** across 14 plant species
- 💊 **Treatment & prevention advice** for every detected disease
- 📊 **Top-5 predictions** with confidence bar chart
- 🗂️ **Prediction history** — browse all past scans in a table
- 📄 **Detail report page** for each individual prediction
- 🖥️ **Responsive UI** — works on desktop, tablet, and mobile
- ⚡ **Demo mode** — fully functional UI even without a trained model

---

## 🗂️ Project Structure

```
leafscan/
│
├── manage.py                         # Django management entry point
├── requirements.txt                  # Python dependencies
├── train_model.py                    # Model training script
├── README.md                         # This file
│
├── plant_disease_project/            # Django project configuration
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
│
├── detector/                         # Main Django application
│   ├── __init__.py
│   ├── admin.py                      # Django admin configuration
│   ├── apps.py
│   ├── models.py                     # PredictionHistory database model
│   ├── views.py                      # Upload, predict, history, detail, about
│   ├── urls.py                       # URL routing
│   ├── ml_utils.py                   # ML inference engine + disease info DB
│   │
│   ├── ml_model/
│   │   └── plant_disease_model.h5    ← Place your trained model HERE
│   │
│   ├── migrations/
│   │   └── __init__.py
│   │
│   └── templates/detector/
│       ├── base.html                 # Shared nav/footer layout
│       ├── index.html                # Home page with upload widget
│       ├── history.html              # Prediction history table
│       ├── detail.html               # Single prediction detail & report
│       └── about.html                # About page with dataset info
│
├── media/                            # Uploaded images (auto-created)
│   └── uploads/
│
└── dataset/                          # Place Kaggle dataset here for training
    ├── train/
    └── valid/
```

---

## 🛠️ Tech Stack

| Layer            | Technology                                        |
|------------------|---------------------------------------------------|
| **Backend**      | Django 4.x (Python)                               |
| **ML / AI**      | TensorFlow 2.x / Keras — ResNet50 fine-tuned CNN  |
| **Frontend**     | HTML5, CSS3, Vanilla JavaScript (no framework)    |
| **Database**     | SQLite (built-in, zero configuration)             |
| **Image Proc.**  | Pillow, NumPy                                     |
| **Fonts / Icons**| Google Fonts (DM Sans, Playfair Display), Font Awesome 6 |

---

## ✅ Prerequisites

| Requirement   | Version          |
|---------------|------------------|
| Python        | 3.9 – 3.11       |
| pip           | latest           |
| Git (optional)| any              |

> **GPU (optional, for faster training):** Install the CUDA-compatible build of TensorFlow. Inference runs fine on CPU.

---

## 🚀 Installation & Setup

### Step 1 — Get the Project

```bash
# Option A — from downloaded ZIP
unzip leafscan.zip
cd leafscan

# Option B — from git
git clone <repo-url>
cd leafscan
```

---

### Step 2 — Create a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

---

### Step 3 — Install Python Dependencies

```bash
pip install -r requirements.txt
```

> **TensorFlow-free install** (for demo mode only, no real ML inference):
> ```bash
> pip install Django Pillow numpy
> ```

---

### Step 4 — Apply Database Migrations

```bash
python manage.py makemigrations
python manage.py migrate
```

---

### Step 5 — (Optional) Create a Superuser

Allows access to the Django admin panel at `/admin/`.

```bash
python manage.py createsuperuser
```

---

### Step 6 — Start the Development Server

```bash
python manage.py runserver
```

Open your browser: **http://127.0.0.1:8000**

---

## 🧠 Training the Model

### Step 1 — Download the Dataset from Kaggle

1. Visit: [https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
2. Click **Download** (free Kaggle account required)
3. Extract the ZIP so your folder structure looks exactly like this:

```
leafscan/
└── dataset/
    ├── train/
    │   ├── Apple___Apple_scab/         (images here)
    │   ├── Apple___Black_rot/
    │   ├── Apple___Cedar_apple_rust/
    │   ├── Apple___healthy/
    │   ├── ... (38 class folders total)
    └── valid/
        ├── Apple___Apple_scab/
        ├── Apple___Black_rot/
        └── ... (38 class folders total)
```

---

### Step 2 — Run the Training Script

```bash
# Default settings: 15 epochs, batch size 32
python train_model.py --dataset_path ./dataset

# Custom settings
python train_model.py --dataset_path ./dataset --epochs 20 --batch_size 64

# All available options
python train_model.py --help
```

**Training runs in two phases:**

| Phase   | Epochs        | What happens                                        |
|---------|---------------|-----------------------------------------------------|
| Phase 1 | First 5       | Only the classification head is trained; ResNet50 backbone is frozen (fast) |
| Phase 2 | Remaining     | Top 30 ResNet50 layers are unfrozen for fine-tuning (best accuracy) |

The **best-performing model** is saved automatically to:
```
detector/ml_model/plant_disease_model.h5
```

---

### Step 3 — Restart the Server

```bash
python manage.py runserver
```

The app will now load and use your trained model for real predictions.

---

## 🖥️ Running the Application

| URL                                  | Page Description                            |
|--------------------------------------|---------------------------------------------|
| `http://127.0.0.1:8000/`            | Home page — drag-and-drop leaf image upload |
| `http://127.0.0.1:8000/history/`    | Prediction history table                    |
| `http://127.0.0.1:8000/prediction/<id>/` | Full report for a single prediction    |
| `http://127.0.0.1:8000/about/`      | About page — dataset & model info           |
| `http://127.0.0.1:8000/admin/`      | Django admin panel (superuser only)         |

---

## 📦 Dataset Details

**New Plant Diseases Dataset** — curated by Samir Bhattarai  
🔗 https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset

| Stat                | Value              |
|---------------------|--------------------|
| Total Images        | 87,000+            |
| Disease Classes     | 38                 |
| Plant Species       | 14                 |
| Train / Valid Split | 80% / 20%          |
| Image Format        | JPG (RGB)          |
| Model Input Size    | 224 × 224 pixels   |

---

## 🌱 Supported Plants & Diseases (38 Classes)

| Plant          | Conditions Detected                                                                                         |
|----------------|-------------------------------------------------------------------------------------------------------------|
| Apple          | Apple Scab, Black Rot, Cedar Apple Rust, Healthy                                                            |
| Blueberry      | Healthy                                                                                                     |
| Cherry         | Powdery Mildew, Healthy                                                                                     |
| Corn (Maize)   | Gray Leaf Spot, Common Rust, Northern Leaf Blight, Healthy                                                  |
| Grape          | Black Rot, Esca (Black Measles), Leaf Blight (Isariopsis), Healthy                                         |
| Orange         | Huanglongbing (Citrus Greening)                                                                              |
| Peach          | Bacterial Spot, Healthy                                                                                     |
| Bell Pepper    | Bacterial Spot, Healthy                                                                                     |
| Potato         | Early Blight, Late Blight, Healthy                                                                          |
| Raspberry      | Healthy                                                                                                     |
| Soybean        | Healthy                                                                                                     |
| Squash         | Powdery Mildew                                                                                              |
| Strawberry     | Leaf Scorch, Healthy                                                                                        |
| Tomato         | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |

---

## ⚙️ How the System Works

```
[User] Upload leaf image (JPG/PNG/WebP)
          ↓
[Django view] Save image → media/uploads/
          ↓
[ml_utils.py] predict_disease(image_path)
    1. PIL opens image → convert to RGB
    2. Resize to 224 × 224
    3. Normalise pixels to [0.0, 1.0]
    4. Add batch dimension → shape (1, 224, 224, 3)
          ↓
[TensorFlow/Keras] ResNet50 model.predict()
    → softmax probabilities for all 38 classes
          ↓
[ml_utils.py] Extract top class + confidence
    → Look up DISEASE_INFO dict for treatment/prevention
    → Build top-5 prediction list
          ↓
[Django view] Save PredictionHistory to SQLite
    → Return JSON to browser
          ↓
[JavaScript] Render result card in the UI
```

---

## ⚡ Demo Mode

If no trained model file is found at `detector/ml_model/plant_disease_model.h5`, the app automatically enters **Demo Mode**:

- A deterministic pseudo-random softmax output is generated, seeded from the image pixel values
- Every page of the UI works correctly — upload, result card, history, detail report
- A yellow info banner is shown on the result card to indicate that predictions are illustrative
- To switch to real predictions: train the model with `train_model.py` and place the `.h5` file at the path above

---

## 📝 Notes for Internship Submission

- All **38 disease classes** from the New Plant Diseases Dataset are implemented
- **Treatment, prevention, and symptom information** is provided for every class
- The codebase follows **Django best practices**: separated app, models, views, templates, URLs
- The **ML pipeline is modular** — swap the model file at any time without touching application code
- **SQLite** requires no external database setup, making the project easy to run on any machine
- The UI is **fully responsive** and built with pure HTML/CSS/JavaScript — no frontend framework required

---

## 📄 License

Developed as part of a remote internship project.  
The dataset is subject to its own [Kaggle licence](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset).
