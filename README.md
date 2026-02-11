# 🏦 TrustFlow: Loan Approval Prediction System

**TrustFlow** is a high-precision Machine Learning application designed to automate credit risk assessment. By leveraging a Decision Tree algorithm, the system evaluates applicant data to provide instant, data-driven loan approval decisions.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://trustflow-creditwise-loanapprovalsystem-tbd3rcuwd279azk2yidr5n.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)

---

## 🚀 Live Demo
You can try the live application here: **[Click to Open TrustFlow App](https://trustflow-creditwise-loanapprovalsystem-tbd3rcuwd279azk2yidr5n.streamlit.app/)**

---

## 📋 Project Overview
The primary goal of this project was to build a model that prioritizes **Precision**. In the banking sector, approving a loan for a high-risk applicant (False Positive) is more costly than rejecting a potentially good one. This system is tuned to minimize that financial risk.

### Key Features:
* **Multi-Dimensional Analysis:** Processes 27 unique features per applicant.
* **Advanced Feature Engineering:** Includes polynomial features ($DTI^2$ and $Credit Score^2$) to capture non-linear risks.
* **Interactive UI:** A clean, responsive web interface built with Streamlit for easy data entry.

---

## 🛠️ Tech Stack
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)

---

## 📊 How the Model Works (The Pipeline)
The model follows a rigorous data science pipeline as documented in the `TrustFlowLoanSystem.ipynb`:

1. **Data Cleaning:**
   - Handled missing values using statistical imputation (Mean for numerical, Mode for categorical).
   - Removed outliers to ensure robust model training.
2. **Encoding:**
   - Converted categorical data (Property Area, Gender, Marital Status, Employment, etc.) into numerical format using **One-Hot Encoding**.
3. **Scaling:**
   - Applied `StandardScaler` to ensure features like `Income` and `Credit Score` are evaluated on the same mathematical scale.
4. **Logic:**
   - The **Decision Tree** algorithm creates a series of hierarchical "if-then" rules based on historical patterns to reach a final classification (Approved vs. Rejected).



---

## 📈 Model Evaluation
The model was evaluated using a test split from the `loan_approval_data.csv`. The Decision Tree was selected as the best model based on its superior **Precision** score.

| Metric | Score | Importance |
| :--- | :--- | :--- |
| **Accuracy** | **85%** | High overall predictive reliability. |
| **Precision** | **89%** | Optimized to minimize False Positives (high-risk approvals). |
| **Recall** | **78%** | Ability to identify eligible candidates correctly. |
| **F1-Score** | **0.83** | Balanced score between precision and recall. |

---

## 📁 File Structure
* `app.py`: The production-ready script for the Streamlit web interface.
* `TrustFlowLoanSystem.ipynb`: The research notebook containing exploratory data analysis (EDA) and model training.
* `loan_model.pkl`: The serialized "brain" of the project (Trained Model).
* `scaler.pkl`: The saved StandardScaler object to normalize user input.
* `requirements.txt`: List of dependencies for cloud deployment.

---

## ✍️ Author
**Ritesh Kumar**
* **GitHub:** [@ritesh-kr18](https://github.com/ritesh-kr18)
* **Project Repository:** [TrustFlow-Credit_Wise-Loan_Approval_System](https://github.com/ritesh-kr18/TrustFlow-Credit_Wise-Loan_Approval_System)

---

## ⚙️ Setup & Installation
To run this project on your local machine:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/ritesh-kr18/TrustFlow-Credit_Wise-Loan_Approval_System.git](https://github.com/ritesh-kr18/TrustFlow-Credit_Wise-Loan_Approval_System.git)
   cd TrustFlow-Credit_Wise-Loan_Approval_System
