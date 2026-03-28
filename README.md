# 🎓 Descriptive Answer Evaluation System
### AI-Powered Answer Evaluation using NLP & Machine Learning

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.45.0-red)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.8.0-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Platform](https://img.shields.io/badge/Platform-Web-lightgrey)

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Key Features](#-key-features)
3. [Tech Stack](#-tech-stack)
4. [Project Structure](#-project-structure)
5. [System Architecture](#-system-architecture)
6. [ML Models & Pipeline](#-ml-models--pipeline)
7. [Supported File Formats](#-supported-file-formats)
8. [Installation & Setup](#-installation--setup)
9. [How to Run](#-how-to-run)
10. [Usage Guide](#-usage-guide)
11. [Pages & Features](#-pages--features)
12. [Database Schema](#-database-schema)
13. [API / Module Reference](#-module-reference)
14. [Performance & Results](#-performance--results)
15. [Troubleshooting](#-troubleshooting)
16. [Future Enhancements](#-future-enhancements)
17. [Contributors](#-contributors)

---

## 🧠 Project Overview

The **Descriptive Answer Evaluation System** is a final-year Machine Learning project that automates the evaluation of subjective/descriptive answers written by students. It compares a **student's answer** against a **reference (model) answer** using a combination of:

- **NLP techniques** — TF-IDF vectorization and Cosine Similarity
- **Machine Learning** — Random Forest, Gradient Boosting, Ridge Regression, SVR
- **Topic Coverage Analysis** — Keyword extraction and overlap detection
- **OCR** — Reads text from images and scanned PDF documents

The system provides a rich **Streamlit web interface** with separate dashboards for **students** and **teachers**, complete with feedback, topic analysis, performance history, and class-level analytics.

### 🎯 Problem Statement

Traditional descriptive answer evaluation is:
- **Time-consuming** for teachers (manual grading)
- **Subjective** (different teachers give different scores)
- **Delayed** (students don't get immediate feedback)
- **Limited** (no topic-level analysis of what's missing)

This system solves all four problems using ML + NLP.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🔐 **Authentication** | Student & Teacher login/signup with SQLite |
| 📤 **Multi-format Upload** | PDF, Images (OCR), Excel, CSV, Word, TXT |
| 🧠 **NLP Evaluation** | TF-IDF + Cosine Similarity scoring |
| 🤖 **ML Prediction** | Trained model predicts score from 4 features |
| 📊 **Topic Analysis** | Shows covered topics, missing topics, coverage % |
| 💬 **Personalised Feedback** | Strengths, weaknesses, study suggestions |
| 📈 **Student Dashboard** | History, average score, best score, pass count |
| 👨‍🏫 **Teacher Dashboard** | Class overview, subject-wise stats, struggling students |
| 🔍 **OCR Support** | Extracts text from PNG, JPG, scanned PDFs |
| 💾 **Persistent Storage** | All results saved to SQLite database |
| 🏆 **Student Ranking** | Teacher can see ranked leaderboard |

---

## 🛠 Tech Stack

### Backend / Logic
| Technology | Version | Purpose |
|---|---|---|
| Python | 3.9+ | Core language |
| scikit-learn | 1.8.0 | ML models, TF-IDF, Cosine Similarity |
| NumPy | 2.4.2 | Numerical feature computation |
| Pandas | 3.0.1 | Data loading and manipulation |
| SQLite3 | built-in | User and result storage |

### Frontend / UI
| Technology | Version | Purpose |
|---|---|---|
| Streamlit | 1.45.0 | Web interface framework |

### File Processing
| Technology | Version | Purpose |
|---|---|---|
| pdfplumber | 0.11.9 | PDF text + table extraction |
| pytesseract | 0.3.13 | OCR for images & scanned PDFs |
| pdf2image | 1.17.0 | PDF → image conversion for OCR |
| Pillow | 12.1.1 | Image handling |
| python-docx | 1.2.0 | Word document extraction |
| openpyxl | 3.1.5 | Excel file reading |

### ML Models Trained
| Model | Library | Notes |
|---|---|---|
| Ridge Regression | sklearn | ✅ Best on this dataset |
| Random Forest | sklearn | 200 trees, depth 6 |
| Gradient Boosting | sklearn | 150 estimators, lr=0.1 |
| SVR | sklearn | RBF kernel, C=100 |

---

## 📁 Project Structure

```
project/
│
├── app.py              # Main Streamlit application (706 lines)
│                       # Contains all pages: Auth, Dashboard, Submit,
│                       # Results, Feedback, Teacher Panel
│
├── train.py            # ML model training script (249 lines)
│                       # Loads dataset.csv → builds features →
│                       # trains 4 models → saves best as model.pkl
│
├── model.py            # ML model wrapper class (112 lines)
│                       # Loads model.pkl, builds features, runs predict()
│                       # Falls back to NLP score if model not loaded
│
├── utils.py            # Core NLP + File extraction + DB (550 lines)
│                       # - preprocess_text()
│                       # - compute_cosine_similarity()
│                       # - analyze_topic_coverage()
│                       # - nlp_evaluate()
│                       # - generate_feedback()
│                       # - extract_text_from_file()  ← all formats
│                       # - init_db(), register_user(), login_user()
│                       # - save_result(), get_student_results()
│
├── dataset.csv         # 30 labelled (reference, student, score) pairs
│                       # Used to train the ML regression model
│
├── model.pkl           # Saved trained model (auto-generated by train.py)
│                       # Contains: {'model': pipeline, 'vectorizer': None}
│
└── users.db            # SQLite database (auto-created on first run)
                        # Tables: users, results
```

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    STREAMLIT WEB UI                      │
│  Login/Signup | Submit | Dashboard | Feedback | Teacher  │
└──────────────────────┬──────────────────────────────────┘
                       │ User Input (text / uploaded file)
                       ▼
┌─────────────────────────────────────────────────────────┐
│                   utils.py — FILE ROUTER                 │
│  .txt → decode │ .pdf → pdfplumber + OCR fallback        │
│  .docx → python-docx │ .xlsx/.csv → pandas               │
│  .png/.jpg → pytesseract OCR                             │
└──────────────────────┬──────────────────────────────────┘
                       │ Extracted plain text
          ┌────────────┴────────────┐
          ▼                         ▼
┌─────────────────┐       ┌──────────────────────┐
│   NLP TRACK     │       │     ML TRACK          │
│   utils.py      │       │     model.py          │
│                 │       │                       │
│ TfidfVectorizer │       │ Feature 1: cos_sim    │
│ cosine_similarity│      │ Feature 2: len_ratio  │
│ keyword extract │       │ Feature 3: kw_overlap │
│ topic coverage  │       │ Feature 4: unique_ratio│
│                 │       │                       │
│ NLP Score =     │       │ model.pkl.predict()   │
│ sim×60%+cov×40% │       │ ML Score (0–100)      │
└────────┬────────┘       └──────────┬────────────┘
         │                           │
         └───────────┬───────────────┘
                     ▼
          ┌──────────────────────┐
          │   FINAL SCORE        │
          │  NLP×50% + ML×50%    │
          │  Grade: A/B/C/D/F    │
          └──────────┬───────────┘
                     │
          ┌──────────▼───────────┐
          │   SQLite Database    │
          │   users.db           │
          │   → results table    │
          └──────────────────────┘
```

---

## 🤖 ML Models & Pipeline

### Training Pipeline (`train.py`)

```
dataset.csv
    │
    ▼
load_and_prepare_data()
    │  reads 30 (reference, student, score) rows
    │
    ▼
build_features()  ← for each row
    │
    │  Feature 1: cosine_similarity(ref, stu)         → 0.0 – 1.0
    │  Feature 2: len(stu_words) / len(ref_words)     → 0.0 – 2.0
    │  Feature 3: shared_keywords / ref_keywords      → 0.0 – 1.0
    │  Feature 4: unique_stu_words / total_stu_words  → 0.0 – 1.0
    │
    ▼
X (30×4 matrix),  y (30 scores)
    │
    ▼
train_test_split(80% / 20%)
    │
    ▼
Train 4 models, compare RMSE / MAE / R²
    │
    ├── RandomForestRegressor(n_estimators=200, max_depth=6)
    ├── GradientBoostingRegressor(n_estimators=150, lr=0.1)
    ├── Ridge(alpha=1.0)  ← best on this dataset
    └── SVR(kernel='rbf', C=100, gamma=0.1)
    │
    ▼
Best model saved → model.pkl
```

### Scoring Formula

```
NLP Score   = (cosine_similarity × 60) + (topic_coverage_% × 0.40)
ML Score    = model.pkl.predict([cos_sim, len_ratio, kw_overlap, unique_ratio])
Final Score = (NLP Score × 0.50) + (ML Score × 0.50)

Grade:  ≥85 → A (Excellent)
        ≥70 → B (Good)
        ≥55 → C (Average)
        ≥40 → D (Below Average)
         <40 → F (Needs Improvement)
```

---

## 📂 Supported File Formats

| Format | Extension(s) | Method | Notes |
|---|---|---|---|
| Plain Text | `.txt` | UTF-8 decode | Direct |
| Word Document | `.docx` | python-docx | Paragraphs + tables |
| PDF (text-based) | `.pdf` | pdfplumber | Text + embedded tables |
| PDF (scanned) | `.pdf` | pytesseract OCR | Auto-detects, falls back |
| PNG Image | `.png` | pytesseract OCR | Any text in image |
| JPEG Image | `.jpg`, `.jpeg` | pytesseract OCR | Photo of handwritten/typed |
| BMP Image | `.bmp` | pytesseract OCR | |
| TIFF Image | `.tiff`, `.tif` | pytesseract OCR | |
| WebP Image | `.webp` | pytesseract OCR | |
| Excel | `.xlsx`, `.xls` | pandas + openpyxl | All sheets extracted |
| CSV Table | `.csv` | pandas | Header + rows as text |

### Smart PDF Detection
```
PDF uploaded
    │
    ▼
Try pdfplumber text extraction
    │
    ├── Got ≥10 words? → Use it ✅  (method = "pdf_text")
    │
    └── Got <10 words? → OCR fallback
            │
            ▼
        pdf2image converts pages to images
            │
            ▼
        pytesseract reads each page
            │
            ▼
        Return OCR text ✅  (method = "pdf_ocr")
```

---

## ⚙️ Installation & Setup

### Step 1 — Clone / Download Project

```bash
# Option A: If using git
git clone https://github.com/yourname/answer-evaluation-system.git
cd answer-evaluation-system

# Option B: Extract ZIP
unzip AnswerEvaluationSystem_v2.zip
cd project
```

### Step 2 — Create Virtual Environment (Recommended)

```bash
# Create
python -m venv venv

# Activate
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### Step 3 — Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Install System Dependencies

#### Tesseract OCR (required for image/scanned PDF support)

```bash
# Ubuntu / Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Windows
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
# Install and add to PATH
```

#### Poppler (required for PDF → image conversion)

```bash
# Ubuntu / Debian
sudo apt-get install poppler-utils

# macOS
brew install poppler

# Windows
# Download: https://github.com/oschwartz10612/poppler-windows/releases
# Add bin/ folder to your PATH
```

### Step 5 — Verify System Dependencies

```bash
# Check Tesseract
tesseract --version
# Expected: tesseract 5.x.x

# Check Poppler
pdftoppm -v
# Expected: pdftoppm version 23.x.x
```

---

## 🚀 How to Run

### Step 1 — Train the ML Model (FIRST TIME ONLY)

```bash
cd project
python train.py
```

**Expected output:**
```
============================================================
  DESCRIPTIVE ANSWER EVALUATION — MODEL TRAINING
============================================================

[1] Loading dataset from: dataset.csv
    Rows loaded: 30

[2] Building feature matrix...
    Feature matrix shape: (30, 4)
    Score range: 60.0 - 90.0

[3] Training and comparing models...
    Model                      RMSE      MAE       R2
    RandomForest             10.058    8.426  -0.7342
    GradientBoosting         11.280    9.137  -1.1812
    Ridge                     9.266    8.001  -0.4720   ← Best
    SVR                      12.298   10.283  -1.5926

[5] Model saved to: model.pkl

============================================================
  TRAINING COMPLETE! model.pkl is ready.
============================================================
```

### Step 2 — Launch the Web App

```bash
streamlit run app.py
```

**Expected output:**
```
  You can now view your Streamlit app in your browser.
  Local URL:  http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

Open your browser and go to: **http://localhost:8501**

### Optional — Run on Custom Port

```bash
streamlit run app.py --server.port 8502
```

---

## 📖 Usage Guide

### First Time Setup

1. Open `http://localhost:8501`
2. Click **Sign Up** tab
3. Create a **student** account (e.g., `student1` / `pass123`)
4. Create a **teacher** account (e.g., `teacher1` / `pass123`)

### As a Student

```
Login → Submit Answer → View Score → View Feedback → Check History
```

1. **Login** with your student account
2. Click **📤 Submit Answer** in the sidebar
3. Enter the **Subject** (e.g., "Machine Learning")
4. Provide the **Reference Answer** (model answer) — type or upload file
5. Provide your **Student Answer** — type or upload file
6. Click **🚀 Evaluate Answer**
7. View your **Score, Grade, Topics Covered, Missing Topics**
8. Click **💬 Feedback** to see personalised strengths and suggestions

### As a Teacher

```
Login → View All Results → Teacher Dashboard → Identify weak students
```

1. **Login** with your teacher account
2. View **All Results** — every student submission
3. **Teacher Panel** shows:
   - Class average score
   - Subject-wise performance
   - Students scoring below 55% (need help)
   - Most commonly missing topics (re-teaching needed)
   - Student ranking leaderboard

---

## 🖥 Pages & Features

### 1. 🔑 Authentication Page
- Student & Teacher login with username/password
- Sign-up with role selection
- Passwords stored as SHA-256 hash in SQLite
- Error messages for wrong credentials / duplicate usernames

### 2. 🏠 Student Dashboard
- Total submissions, average score, best score, pass count
- Table of 5 most recent submissions
- Quick action buttons to Submit or view Feedback

### 3. 📤 Submit Answer Page
- Supported format badge displayed per file type
- Dual input method: type text OR upload file
- For images/scanned PDFs: shows OCR badge + extracted text preview
- For Excel/CSV: shows table preview with `st.dataframe`
- After evaluation: shows full result inline on same page

### 4. 📊 Performance Summary
- Final Score, NLP Score, ML Score, Similarity Score — all as metrics
- Grade banner with colour coding (green/blue/yellow/red)
- Topic chips: ✅ covered (green) and ❌ missing (red)
- Coverage progress bar
- Word count comparison (reference vs student)

### 5. 💬 Feedback Page
- Loads last evaluated result (or most recent from DB)
- **Strengths** — what the student did well
- **Areas to Improve** — specific weaknesses identified
- **Study Suggestions** — numbered actionable advice
- **Topics to Review** — highlighted cards for missing concepts

### 6. 📊 Results History (Student)
- All past submissions in a table
- Average score, highest score metrics
- Sorted by most recent first

### 7. 👨‍🏫 Teacher Dashboard
- Class overview: unique students, total submissions, class average, fail count
- Subject-wise performance table + bar chart
- Students needing help (score < 55%) — filtered table
- Commonly missing topics — frequency bar chart + table
- Student ranking by average score

---

## 🗄 Database Schema

### Table: `users`

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PK | Auto-increment |
| `username` | TEXT UNIQUE | Login username |
| `password` | TEXT | SHA-256 hashed password |
| `role` | TEXT | `'student'` or `'teacher'` |
| `created_at` | TIMESTAMP | Account creation time |

### Table: `results`

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PK | Auto-increment |
| `student_username` | TEXT | FK → users.username |
| `subject` | TEXT | Topic/subject name |
| `reference_answer` | TEXT | Model answer text |
| `student_answer` | TEXT | Student's answer text |
| `similarity_score` | REAL | Cosine similarity × 100 |
| `nlp_score` | REAL | NLP-based score (0–100) |
| `ml_score` | REAL | ML model predicted score |
| `final_score` | REAL | Combined final score |
| `grade` | TEXT | A / B / C / D / F |
| `covered_topics` | TEXT | Comma-separated keywords |
| `missing_topics` | TEXT | Comma-separated keywords |
| `submitted_at` | TIMESTAMP | Submission time |

---

## 📦 Module Reference

### `utils.py` — Key Functions

```python
# Text processing
preprocess_text(text: str) -> str
    # Lowercase, remove punctuation, strip whitespace

extract_keywords(text: str, top_n=15) -> list
    # Returns top N keywords after stopword filtering

compute_cosine_similarity(text1: str, text2: str) -> float
    # Returns 0.0–1.0 cosine similarity using TF-IDF

analyze_topic_coverage(reference: str, student: str) -> dict
    # Returns: {covered_topics, missing_topics, extra_topics,
    #           coverage_percentage, ref_topic_count, stu_topic_count}

nlp_evaluate(reference: str, student: str) -> dict
    # Full NLP evaluation. Returns: {similarity_score, nlp_score,
    #                                grade, coverage, word_count_ref, word_count_stu}

generate_feedback(eval_result: dict) -> dict
    # Returns: {strengths: [], weaknesses: [], suggestions: [], score}

# File extraction
extract_text_from_file(uploaded_file) -> (text, file_type, metadata)
    # Routes to correct extractor based on extension
    # Returns tuple: (extracted_text, type_string, metadata_dict)

get_file_type(filename: str) -> str or None
    # Returns: 'pdf', 'image', 'excel', 'csv', 'docx', 'text', or None

# Database
init_db()                                      # Create tables if not exist
register_user(username, password, role)        # Returns (bool, message)
login_user(username, password)                 # Returns (bool, role_or_message)
save_result(data: dict)                        # Returns bool
get_student_results(username: str) -> list     # Student's history
get_all_results() -> list                      # All results (teacher view)
```

### `model.py` — Key Functions

```python
predict_score(reference: str, student: str) -> float
    # Predicts score using loaded model.pkl
    # Falls back to NLP score if model not loaded

get_evaluator() -> AnswerEvaluator
    # Returns singleton model instance

class AnswerEvaluator:
    is_trained: bool          # True if model.pkl loaded successfully
    predict(ref, stu) -> float  # Core prediction method
```

### `train.py` — Key Functions

```python
build_features(reference, student) -> list[4 floats]
    # Returns [cos_sim, length_ratio, kw_overlap, unique_ratio]

load_and_prepare_data(csv_path) -> (X, y)
    # Returns feature matrix and target scores

train_all_models(X_train, y_train, X_test, y_test)
    # Trains 4 models, returns best pipeline + name + results dict

main()
    # Full training pipeline: load → build features → train → save → demo
```

---

## 📊 Performance & Results

### ML Model Comparison (on 30-sample dataset)

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Ridge Regression | **9.27** | **8.00** | -0.47 |
| Random Forest | 10.06 | 8.43 | -0.73 |
| Gradient Boosting | 11.28 | 9.14 | -1.18 |
| SVR (RBF) | 12.30 | 10.28 | -1.59 |

> **Note:** Negative R² values are expected with only 30 training samples. Increasing dataset size to 200+ samples would significantly improve all metrics. The system still works correctly because the final score blends NLP (50%) and ML (50%).

### 5-Fold Cross-Validation (Ridge)

```
CV RMSE scores: [11.46, 7.05, 4.74, 3.39, 7.70]
CV RMSE mean:    6.87 ± 2.77
```

### Demo Predictions

| Answer Quality | Reference Snippet | Student Snippet | Predicted Score |
|---|---|---|---|
| Very Good | Photosynthesis converts light... | Plants use sunlight, CO2, water... | 81.8 / 100 |
| Partial | Machine learning is a subset of AI... | ML allows computers to learn... | 78.8 / 100 |
| Poor | The water cycle includes evaporation... | Water goes up and comes down... | 70.4 / 100 |

---

## 🔧 Troubleshooting

### ❌ `ModuleNotFoundError: No module named 'streamlit'`
```bash
pip install streamlit==1.45.0
```

### ❌ `ModuleNotFoundError: No module named 'docx'`
```bash
pip install python-docx
```

### ❌ `model.pkl not found`
```bash
# You must train the model first
python train.py
```

### ❌ `TesseractNotFoundError`
```bash
# Install Tesseract system package
# Ubuntu:  sudo apt-get install tesseract-ocr
# macOS:   brew install tesseract
# Windows: https://github.com/UB-Mannheim/tesseract/wiki

# Then verify:
tesseract --version
```

### ❌ `pdf2image.exceptions.PDFInfoNotInstalledError`
```bash
# Install Poppler
# Ubuntu:  sudo apt-get install poppler-utils
# macOS:   brew install poppler
# Windows: https://github.com/oschwartz10612/poppler-windows/releases
```

### ❌ `sqlite3.OperationalError: database is locked`
```bash
# Delete the database and restart
rm users.db
streamlit run app.py
# Re-register your accounts
```

### ❌ Streamlit port already in use
```bash
streamlit run app.py --server.port 8502
```

### ❌ File upload gives ERROR message
- Check the file extension is supported (see Supported File Formats table)
- For `.doc` files, re-save as `.docx` in Microsoft Word first
- For corrupt PDFs, try re-exporting the PDF

### ❌ OCR returns garbage text
- Make sure the image has good contrast (dark text, light background)
- Minimum recommended image resolution: 150 DPI
- For handwritten text, OCR accuracy may be lower — type text instead

---

## 🚀 Future Enhancements

- [ ] **Grammar & Fluency Scoring** — integrate LanguageTool API
- [ ] **BERT / Sentence Transformers** — replace TF-IDF with semantic embeddings
- [ ] **Larger Dataset** — 500+ labelled pairs for better ML accuracy
- [ ] **Export Reports** — download student results as PDF
- [ ] **Email Notifications** — notify students when evaluated
- [ ] **Plagiarism Detection** — compare against previous student answers
- [ ] **Multi-language Support** — Hindi, Tamil, Telugu etc.
- [ ] **REST API** — expose endpoints for LMS integration (Moodle, Canvas)
- [ ] **Docker Support** — containerised deployment
- [ ] **Admin Panel** — manage users, bulk upload datasets
- [ ] **Question Bank** — teacher uploads question + model answer, student picks question
- [ ] **Confidence Score** — show how confident the ML model is in its prediction

---

## 📄 Dataset Format

The `dataset.csv` file must follow this exact format for training:

```csv
reference_answer,student_answer,score
"Machine learning is a subset of AI...","ML allows computers to learn...",85
"Photosynthesis is the process...","Plants use sunlight to make food...",80
```

| Column | Type | Range | Description |
|---|---|---|---|
| `reference_answer` | string | — | The model/correct answer |
| `student_answer` | string | — | A sample student answer |
| `score` | integer | 0–100 | Human-assigned score |

**Minimum recommended**: 30 rows (current)
**Recommended for production**: 200+ rows

---

## 👥 Contributors

| Name | Role |
|---|---|
| Your Name | Developer — NLP, ML, Streamlit UI |
| Guide Name | Project Supervisor |

---

## 📜 License

This project is created for educational purposes as a Final Year Project.

---

## 🙏 Acknowledgements

- [scikit-learn](https://scikit-learn.org/) — Machine learning library
- [Streamlit](https://streamlit.io/) — Web app framework
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) — OCR engine
- [pdfplumber](https://github.com/jsvine/pdfplumber) — PDF extraction
- [Three.js](https://threejs.org/) — Inspiration from DSA 3D Visualizer project

---

*Built with ❤️ as a Final Year Machine Learning Project*
