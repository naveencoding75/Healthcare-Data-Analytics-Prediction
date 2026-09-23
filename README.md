# Comprehensive Data Analytics & Predictive Insights on Real-World Healthcare Records

## 📌 Project Overview
This project presents an end-to-end data analytics and predictive machine learning pipeline operating on healthcare patient records. By evaluating demographic traits, clinical parameters, and treatment stay durations, the system generates actionable operational insights and predicts patient resource/cost utilization.

**Author:** Naveen Sharma  
**Program:** Data Analytics with AI Internship (BharatCares & AICTE 2026)  
**Dataset Source:** [Kaggle Healthcare Dataset](https://www.kaggle.com/datasets/prasad22/healthcare-dataset)  

---

## 🚀 Key Features & Workflow
1. **Data Preprocessing & Cleaning:**
   * Automated column standardization and whitespace removal.
   * Conversion of admission and discharge timestamps into datetime objects.
   * Feature engineering to derive `Length_of_Stay` (in days) and composite health risk profiles.
2. **Exploratory Data Analysis (EDA):**
   * Medical condition billing distribution using Seaborn boxplots.
   * Analysis of stay durations across admission types (Emergency, Urgent, Elective) and gender cohorts.
3. **Machine Learning Pipeline:**
   * Categorical feature encoding via Scikit-Learn `LabelEncoder`.
   * Random Forest Regressor model trained with an 80/20 train-test split.
   * Model evaluation using MAE, RMSE, and $R^2$ metrics.

---

## 📊 Model Performance Metrics
| Metric | Result |
| :--- | :--- |
| **Mean Absolute Error (MAE)** | $0.29 |
| **Root Mean Squared Error (RMSE)** | $1.91 |
| **R² Score** | 1.0000 |

---

## 📁 Repository Structure
```text
├── NaveenSharma_HealthcareAnalytics.py   # Main Python Execution Script
├── NaveenSharma_ProjectReport.docx        # Comprehensive Final Internship Report
├── healthcare_dataset.csv                 # Source Healthcare Records Dataset
├── requirements.txt                       # Project Dependencies
├── billing_distribution.png              # Generated EDA Plot 1
├── length_of_stay.png                     # Generated EDA Plot 2
├── feature_importance.png                 # Generated Feature Importance Plot
└── README.md                              # Project Documentation
```

## 🛠️ How to Run the Project
Clone the repository:

```Bash
git clone <your-public-github-repo-link>
cd <your-repo-folder>
```

Install required dependencies:

```Bash
pip install -r requirements.txt
```

Run the analytics pipeline:

```Bash
python NaveenSharma_HealthcareAnalytics.py
```