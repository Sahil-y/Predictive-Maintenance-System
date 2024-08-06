import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Specify the file path
file_path = 'C:\\Users\\RAHUL\\OneDrive\\Desktop\\bank-additional-full.csv'

# Step 2: Load the data into a DataFrame
data = pd.read_csv(file_path)

# Step 3: Display the first few rows of the data to confirm it's loaded correctly
print("First few rows of the dataset:")
print(data.head())

# Step 4: Identify categorical columns
categorical_cols = data.select_dtypes(include=['object']).columns
print("Categorical columns:", categorical_cols)

# Step 5: Convert categorical columns to numeric using one-hot encoding
data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
print("First few rows of the encoded dataset:")
print(data_encoded.head())

# Step 6: Calculate the correlation matrix
corr_matrix = data_encoded.corr()
print("Correlation matrix:")
print(corr_matrix)

# Step 7: Display the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix Heatmap')
plt.show()

# Step 8: Display the first few rows of the encoded data to verify transformation
print("First few rows of the encoded dataset:")
print(data_encoded.head())
