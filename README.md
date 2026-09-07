# Titanic-survival-prediction
# 🚢 Titanic Dataset Analysis & Delinquency/Survival Prediction

An end-to-end data science project analyzing the historical passenger manifest of the *RMS Titanic*. This repository explores demographic features, performs exploratory data analysis (EDA), engineer key features, and builds predictive machine learning models to identify key drivers of survival.

---

## 📌 Project Overview

The objective of this project is to analyze the factors that influenced passenger survival rates during the Titanic disaster. By processing demographic, socioeconomic, and spatial data, this project answers key analytical questions and trains predictive models to classify whether a given passenger survived.

### Key Questions Addressed:
* How did socioeconomic status (`Pclass`) impact survival odds?
* Was the "Women and Children First" protocol reflected in the data?
* Did family size or traveling alone influence passenger outcomes?
* What role did ticket pricing (`Fare`) and embarkation port play?

---

## 📊 Dataset Description

The dataset used in this project is the classic Titanic dataset sourced from Kaggle. 

| Feature | Description | Type |
| :--- | :--- | :--- |
| `PassengerId` | Unique ID for each passenger | Numerical |
| `Survived` | Survival status (0 = No, 1 = Yes) | Target (Binary) |
| `Pclass` | Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd) | Categorical/Ordinal |
| `Name` | Full name and title | Text |
| `Sex` | Gender (male / female) | Categorical |
| `Age` | Age in years | Numerical (Continuous) |
| `SibSp` | Number of siblings/spouses aboard | Numerical (Discrete) |
| `Parch` | Number of parents/children aboard | Numerical (Discrete) |
| `Ticket` | Ticket number | Text |
| `Fare` | Passenger fare | Numerical (Continuous) |
| `Cabin` | Cabin number | Text |
| `Embarked` | Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton) | Categorical |

---

## 🛠️ Project Workflow

1. **Data Cleaning & Imputation**
   * Handled missing values in `Age` using median imputation grouped by `Pclass` and `Sex`.
   * Imputed missing `Embarked` values using the mode.
   * Addressed missing/sparse `Cabin` data by creating structural indicators.

2. **Feature Engineering**
   * Extracted passenger titles (`Mr`, `Mrs`, `Miss`, `Master`, etc.) from names.
   * Created a `FamilySize` feature (`SibSp` + `Parch` + 1).
   * Created a binary `IsAlone` feature for solo travelers.

3. **Exploratory Data Analysis (EDA)**
   * Univariate and bivariate analysis using `seaborn` and `matplotlib`.
   * Correlation heatmaps to identify strong predictor variables.

4. **Predictive Modeling**
   * Encoded categorical variables using One-Hot and Label Encoding.
   * Scaled numerical features using `StandardScaler`.
   * Trained and evaluated baseline classifiers (e.g., Logistic Regression, Random Forest, Decision Trees).

---

## 💻 Tech Stack & Libraries

* **Language:** Python 3.x
* **Environment:** Jupyter Notebook / Google Colab
* **Data Manipulation:** `pandas`, `numpy`
* **Data Visualization:** `matplotlib`, `seaborn`
* **Machine Learning:** `scikit-learn`

---

## 📁 Repository Structure

```text
.
├── Titanic_Dataset_Analysis.ipynb   # Main Jupyter Notebook with full analysis & code
├── data/                            # Raw dataset files (train.csv, test.csv)
├── README.md                        # Project documentation
└── .gitignore                       # Git ignore file for checkpoints and local cache
