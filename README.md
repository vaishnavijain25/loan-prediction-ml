# 🏦 Loan Prediction ML App

## 📌 Project Aim

The aim of this project is to build an end-to-end Machine Learning application that predicts whether a loan application will be Approved or Rejected based on applicant details.

This project covers:
- Data preprocessing
- Feature engineering
- Machine Learning model training
- Model evaluation
- Streamlit frontend development
- Deployment on Render

---

# 🚀 Live Demo

🔗 Live App:  
https://loan-prediction-ml-fviz.onrender.com/

---

# 📂 GitHub Repository

🔗 Repository Link:  
https://github.com/vaishnavijain25/loan-prediction-ml

---

# 🛠️ Tech Stack

## Programming Language
- Python

## Libraries Used
- Pandas
- NumPy
- Scikit-learn
- Streamlit

## Tools & Platforms
- VS Code
- Git
- GitHub
- Render

---

# 📊 Machine Learning Models Used

The following models were trained and evaluated:

- Logistic Regression
- Random Forest Classifier
- Decision Tree Classifier

🏆 Final Selected Model:
- Random Forest Classifier

---

# 📈 Model Performance

| Model | Accuracy | F1 Score |
|---|---|---|
| Logistic Regression | 91.33% | 0.9311 |
| Random Forest | 98.13% | 0.9850 |
| Decision Tree | 98.13% | 0.9849 |

---

# 🔄 Project Workflow

## Step 1 — Data Collection

- Loaded dataset using Pandas
- Validated dataset structure
- Checked feature consistency

### Module Used

```bash
src/components/data_ingestion.py
```

---

## Step 2 — Data Cleaning

Implemented:
- duplicate removal
- missing value handling
- column cleaning
- string value cleaning
- unnecessary column removal

### Module Used

```bash
src/pipeline/data_cleaning.py
```

---

## Step 3 — Data Transformation

Implemented:
- target encoding
- categorical encoding using LabelEncoder
- train-test split
- feature ordering
- artifact saving

### Artifacts Saved

- encoders.pkl
- feature_columns.pkl

### Module Used

```bash
src/pipeline/data_transformation.py
```

---

## Step 4 — Model Training

Trained and evaluated:
- Logistic Regression
- Random Forest
- Decision Tree

Metrics Used:
- Accuracy
- Precision
- Recall
- F1 Score

### Module Used

```bash
src/pipeline/model_trainer.py
```

---

## Step 5 — Artifact Generation

Saved:
- trained model
- encoders
- scaler
- feature columns

### Artifacts

```bash
model.pkl
encoders.pkl
scaler.pkl
feature_columns.pkl
```

---

## Step 6 — Streamlit Frontend

Built an interactive web application where users can:
- enter applicant details
- predict loan approval
- view approval/rejection result instantly

### Module Used

```bash
streamlit_app.py
```

---

## Step 7 — GitHub Version Control

Implemented:
- Git workflow
- repository management
- version control
- .gitignore handling

### Git Commands Used

```bash
git add .
git commit -m "message"
git push origin main
```

---

## Step 8 — Deployment on Render

Deployed the Streamlit application using Render.

### Deployment Flow

```text
VS Code Project
      ↓
GitHub Repository
      ↓
Render Deployment
      ↓
Live Streamlit App
```

---

# 📁 Project Structure

```bash
loan_prediction/
│
├── artifacts/
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── encoders.pkl
│   └── feature_columns.pkl
│
├── logs/
│
├── src/
│   ├── components/
│   │   └── data_ingestion.py
│   │
│   ├── pipeline/
│   │   ├── data_cleaning.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   └── train_pipeline.py
│   │
│   └── utils/
│       └── logger.py
│
├── streamlit_app.py
├── requirements.txt
├── README.md
└── loan.csv
```

---

# ⚙️ Installation & Setup

## 1️⃣ Clone Repository

```bash
git clone https://github.com/vaishnavijain25/loan-prediction-ml.git
```

---

## 2️⃣ Move into Project Folder

```bash
cd loan-prediction-ml
```

---

## 3️⃣ Create Virtual Environment

```bash
python -m venv venv
```

---

## 4️⃣ Activate Virtual Environment

### Windows

```bash
venv\Scripts\activate
```

---

## 5️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

---

# 🧠 Train the Model

Run:

```bash
python -m src.pipeline.train_pipeline
```

This will:
- clean data
- transform features
- train models
- save artifacts

---

# ▶️ Run Streamlit App

```bash
streamlit run streamlit_app.py
```

---

# 📸 Application Screenshots

## ✅ Loan Approved Prediction

<img width="568" height="765" alt="Screenshot 2026-05-10 185038" src="https://github.com/user-attachments/assets/ed79271f-8523-472e-b927-1d465cf8c9f5" />


---

## ❌ Loan Rejected Prediction

<img width="581" height="772" alt="Screenshot 2026-05-10 185216" src="https://github.com/user-attachments/assets/a67ca05d-faf9-45c1-9b29-c80f621d2dbe" />


---

# 🌐 Deployment

The application is deployed on Render.

🔗 Live App:  
https://loan-prediction-ml-fviz.onrender.com/

---

# 🎯 Learning Outcomes

Through this project, I learned:

✅ End-to-end ML pipeline creation  
✅ Data preprocessing techniques  
✅ Model training & evaluation  
✅ Feature engineering  
✅ Debugging ML applications  
✅ Streamlit frontend development  
✅ Git & GitHub workflow  
✅ Deployment using Render  
✅ Real-world project structuring  

---

# 👩‍💻 Author

Vaishnavi Jain

🔗 GitHub:  
https://github.com/vaishnavijain25

---

# ⭐ Support

If you found this project useful, feel free to:

⭐ Star the repository  
🍴 Fork the project  
📢 Share feedback
