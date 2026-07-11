# Heart Disease Prediction using ANN ❤️

**An Artificial Neural Network model that predicts the likelihood of heart disease from patient clinical data.**

## 📌 Problem Statement

Early detection of heart disease risk can enable timely medical intervention. This project trains an ANN classifier on clinical features (age, cholesterol, blood pressure, etc.) to predict the presence of heart disease.

## ✨ Key Features

- Data preprocessing: handling missing values, feature scaling, encoding
- ANN architecture built with TensorFlow/Keras
- Model training with validation tracking (loss/accuracy curves)
- Evaluation via accuracy, precision, recall, confusion matrix
- Comparison against a baseline model *(e.g. Logistic Regression, if you did this)*

## 🧰 Tech Stack

- **Language:** Python
- **Deep Learning:** TensorFlow / Keras
- **Data Handling:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn

## 📂 Project Structure

```
Heart-Disease-Prediction-using-ANN/
├── data/                  # Dataset (or link to source)
├── notebooks/             # EDA and model training notebooks
├── models/                # Saved trained model
├── requirements.txt
└── README.md
```

## ⚙️ Installation

```bash
git clone https://github.com/anandm032/Heart-Disease-Prediction-using-ANN.git
cd Heart-Disease-Prediction-using-ANN
pip install -r requirements.txt
```

## ▶️ Usage

```bash
jupyter notebook notebooks/heart_disease_ann.ipynb
```


## 📊 Results

 "Achieved 88% test accuracy with an ANN of 2 hidden layers (16, 8 units), outperforming a logistic regression baseline by 6 points")*

## 🗺️ Future Work

- Hyperparameter tuning (layers, units, learning rate)
- Cross-validation for more robust evaluation
- Deploy as a simple web form for risk prediction



## 📚 Dataset

[UCI Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) *(update if you used a different source, e.g. Kaggle)*
