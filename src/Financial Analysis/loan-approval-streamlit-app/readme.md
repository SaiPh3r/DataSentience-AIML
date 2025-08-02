
# 💰 Loan Approval Prediction App

A Streamlit web application that predicts whether a loan application will be approved based on user-inputted details. Built using a **Random Forest Classifier** trained on a **SMOTE-balanced dataset** to handle class imbalance.

---

## 📂 Project Structure

```
loan-approval-streamlit-app/
│
├── app.py                  # Streamlit frontend application
├── loan_approval.ipynb     # EDA and model training notebook
├── smote_rf_model.pkl      # Trained Random Forest model with SMOTE
├── readme.md               # Project README
├── data/
│   └── train.csv           # Original training data
└── .gitignore
```

---

## 🧠 Features Used

The model is trained on the following features:

1. `Married` (Yes=1, No=0)  
2. `Dependents` (0, 1, 2, 3)  
3. `Education` (Graduate=1, Not Graduate=0)  
4. `Self_Employed` (Yes=1, No=0)  
5. `ApplicantIncome`  
6. `CoapplicantIncome`  
7. `LoanAmount`  
8. `Loan_Amount_Term`  
9. `Credit_History` (1 = Good, 0 = Bad)  
10. `Property_Urban` (Urban = 1, Rural = 0)

---

## 🧪 Dataset

We used the publicly available [Loan Prediction Problem Dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset?select=train_u6lujuX_CVtuZ9i.csv) from Kaggle.

---

## 🚀 How to Run

1. **Clone the Repository**

(clone the repo)
```bash
cd loan-approval-streamlit-app
```

2. **(Optional) Create a Virtual Environment**

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows use `.venv\Scripts\activate`
```

3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not present, install manually:

```bash
pip install pandas scikit-learn imbalanced-learn streamlit
```

4. **Run the Streamlit App**

```bash
streamlit run app.py
```

---

## 📌 Future Improvements

- Add visualizations to the Streamlit app
- Implement hyperparameter tuning
- Save user submissions to a database
- Dockerize the app for containerized deployment

---

## 👨‍💻 Author

Made with ❤️ by [Sai Anand](https://github.com/saianand857)
