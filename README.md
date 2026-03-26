# 🛡️ HateGuard — Real-Time Hate Speech Detection System

An end-to-end NLP system for real-time hate speech detection with **95% accuracy**, built with Python, scikit-learn, NLTK, Flask/FastAPI, React.js, and Chart.js.

## 🎯 Features

- **24,783 real tweets** from Kaggle hate speech dataset
- **ETL Pipeline**: Text cleaning, NLTK lemmatization, TF-IDF vectorization, meta feature engineering
- **7 ML Models**: Logistic Regression, Random Forest, Gradient Boosting, SVM, MLP, Deep Neural Net, Ensemble
- **Best Model**: Ensemble — 95.13% accuracy, 96% precision, 98% recall
- **Flask REST API** with <120ms latency
- **FastAPI** with auto-generated docs (`/docs`)
- **React.js + Chart.js** analytics dashboard
- **Anonymous Chatbot UI** with real-time hate detection
- **MongoDB Atlas** cloud database for prediction storage
- **Docker-ready** deployment

## 🏗️ Architecture
```
User Message → API → Text Cleaning → NLTK Lemmatization → TF-IDF
→ ML Model (Ensemble) → Hate/Safe Classification → Dashboard + MongoDB
```

## 📊 Model Performance

| Model | Accuracy | F1 Score | AUC-ROC |
|-------|----------|----------|---------|
| Logistic Regression | 94.65% | 96.80% | 97.60% |
| Random Forest | 93.81% | 96.50% | 97.32% |
| Gradient Boosting | 93.04% | 96.00% | 96.72% |
| Linear SVC | 94.65% | 96.80% | 97.60% |
| MLP Neural Net | 92.88% | 95.77% | 96.00% |
| Deep Neural Net (5-layer) | 93.68% | 96.22% | 96.85% |
| **Ensemble (Best)** | **95.13%** | **97.09%** | **98.59%** |

## 🚀 Quick Start
```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/HateSpeechDetection.git
cd HateSpeechDetection

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Download dataset from Kaggle
# https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset
# Place labeled_data.csv in data/ folder

# Run pipeline
python data/convert_kaggle.py
python data/etl_pipeline.py
python models/train_models.py

# Start API
python api/app.py

# Open dashboard
# Open dashboard/index.html in browser
```

## 📁 Project Structure
```
HateSpeechDetection/
├── api/
│   ├── app.py              # Flask API
│   └── main.py             # FastAPI (upgraded)
├── dashboard/
│   ├── index.html          # Analytics dashboard
│   └── chatbot.html        # Anonymous chatbot UI
├── data/
│   ├── generate_dataset.py # Synthetic data generator
│   ├── convert_kaggle.py   # Kaggle dataset converter
│   └── etl_pipeline.py     # ETL pipeline (NLTK NLP)
├── models/
│   ├── train_models.py     # Train 6 ML models
│   └── train_deep_learning.py  # Deep neural network
├── requirements.txt
├── Dockerfile
└── README.md
```

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/predict` | Classify single text |
| POST | `/api/predict/batch` | Classify up to 100 texts |
| GET | `/api/health` | Health check |
| GET | `/api/metrics` | Model performance metrics |
| GET | `/api/stats` | Request statistics |

## 🛠️ Tech Stack

- **Backend**: Python, Flask, FastAPI, Gunicorn
- **ML/NLP**: Scikit-learn, NLTK, TF-IDF, Ensemble Learning
- **Frontend**: HTML, CSS, JavaScript, React.js, Chart.js
- **Database**: MongoDB Atlas
- **Deployment**: Docker, Docker Compose


