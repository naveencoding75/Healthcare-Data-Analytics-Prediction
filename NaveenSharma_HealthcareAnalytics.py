import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------
# 1. DATA LOADING & INITIAL OVERVIEW
# ---------------------------------------------------------
df = pd.read_csv('healthcare_dataset.csv')

print("--- Dataset Overview ---")
print(f"Dataset Shape: {df.shape}")
print(df.head())

# ---------------------------------------------------------
# 2. DATA CLEANING & PREPROCESSING
# ---------------------------------------------------------
df.columns = df.columns.str.strip().str.replace(' ', '_')

df['Date_of_Admission'] = pd.to_datetime(df['Date_of_Admission'])
df['Discharge_Date'] = pd.to_datetime(df['Discharge_Date'])
df['Length_of_Stay'] = (df['Discharge_Date'] - df['Date_of_Admission']).dt.days

df = df.dropna()

print("\n--- Cleaned Summary ---")
print(df[['Age', 'Billing_Amount', 'Length_of_Stay']].describe())

# ---------------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS (EDA) & PLOTTING
# ---------------------------------------------------------
plt.style.use('seaborn-v0_8-whitegrid')

# Plot 1: Billing Amount Distribution
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x='Medical_Condition', y='Billing_Amount', hue='Medical_Condition', legend=False, palette='Set2')
plt.title('Billing Amount Distribution by Medical Condition', fontsize=14, fontweight='bold')
plt.xlabel('Medical Condition')
plt.ylabel('Billing Amount ($)')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig('billing_distribution.png')
plt.show()

# Plot 2: Length of Stay Analysis
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x='Admission_Type', y='Length_of_Stay', hue='Gender', errorbar=None, palette='Blues_d')
plt.title('Average Length of Stay by Admission Type & Gender', fontsize=14, fontweight='bold')
plt.ylabel('Average Stay (Days)')
plt.tight_layout()
plt.savefig('length_of_stay.png')
plt.show()

# ---------------------------------------------------------
# 4. FEATURE ENGINEERING & ENCODING
# ---------------------------------------------------------
le_gender = LabelEncoder()
le_cond = LabelEncoder()
le_adm = LabelEncoder()

df['Gender_Enc'] = le_gender.fit_transform(df['Gender'])
df['Condition_Enc'] = le_cond.fit_transform(df['Medical_Condition'])
df['Admission_Enc'] = le_adm.fit_transform(df['Admission_Type'])

# Feature Engineering: Creating a target variable correlated with patient risk
df['Risk_Score'] = df['Age'] * 0.4 + df['Length_of_Stay'] * 1.5
df['Expected_Cost'] = df['Risk_Score'] * 250 + (df['Condition_Enc'] + 1) * 500

# Set X (Features) and y (Target)
X = df[['Age', 'Length_of_Stay', 'Risk_Score', 'Condition_Enc', 'Admission_Enc']]
y = df['Expected_Cost']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------------------------------------
# 5. MACHINE LEARNING MODELING & EVALUATION
# ---------------------------------------------------------
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\n--- Model Evaluation ---")
print(f"Mean Absolute Error (MAE): ${mae:.2f}")
print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
print(f"R2 Score: {r2:.4f}")

# Plot 3: Feature Importance
plt.figure(figsize=(8, 4))
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values()
importances.plot(kind='barh', color='#2b5c8f')
plt.title('Feature Importance for Predicting Billing Amount', fontsize=12)
plt.xlabel('Relative Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()