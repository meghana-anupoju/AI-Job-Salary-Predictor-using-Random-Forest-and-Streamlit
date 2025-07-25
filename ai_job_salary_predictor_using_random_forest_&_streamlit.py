# -*- coding: utf-8 -*-
"""AI Job Salary Predictor using Random Forest & Streamlit.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1nC6lyBJgKB9TvHPVjNLQo1bwhrwcUyE4
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv("/content/ai_job_dataset.csv")

data.head(10)

data.tail(5)

data.shape

data.info()

data.isnull().sum()

data.describe()

print(data['job_title'].value_counts())

print(data['experience_level'].value_counts())

print(data['employment_type'].value_counts())

print(data['company_location'].value_counts())

print(data['company_size'].value_counts())

plt.figure(figsize=(10, 6))
sns.histplot(data['salary_usd'], bins=30, kde=True)
plt.title('Distribution of Salaries in USD')
plt.xlabel('Salary (USD)')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(12, 8))
sns.boxplot(x='experience_level', y='salary_usd', data=data)
plt.title('Salary Distribution by Experience Level')
plt.xlabel('Experience Level')
plt.ylabel('Salary (USD)')
plt.show()

top_10_jobs = data.groupby('job_title')['salary_usd'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(12, 8))
sns.barplot(x=top_10_jobs.values, y=top_10_jobs.index)
plt.title('Top 10 Highest Paying AI Job Titles')
plt.xlabel('Average Salary (USD)')
plt.ylabel('Job Title')
plt.show()

# Select categorical columns for one-hot encoding
categorical_cols = ['job_title', 'experience_level', 'employment_type', 'company_location', 'company_size']

# Apply one-hot encoding
data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# Display the first 5 rows of the new encoded data
data_encoded.head()

# Define features (X) and target (y)
X = data_encoded.drop(['salary_usd', 'job_id', 'salary_currency', 'employee_residence', 'required_skills', 'education_required', 'industry', 'posting_date', 'application_deadline', 'company_name'], axis=1)
y = data_encoded['salary_usd']

from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestRegressor

# Initialize the model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

# Train the model
rf_model.fit(X_train, y_train)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Make predictions on the test data
y_pred = rf_model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
print(f"Mean Squared Error (MSE): ${mse:,.2f}")
print(f"R-squared (R²): {r2:.2f}")

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
plt.title('Actual vs. Predicted Salaries')
plt.xlabel('Actual Salary (USD)')
plt.ylabel('Predicted Salary (USD)')
plt.show()

# Get feature importances from the trained model
importances = rf_model.feature_importances_
feature_names = X_train.columns

# Create a DataFrame for visualization
feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})

# Sort features by importance and select the top 15
top_15_features = feature_importance_df.sort_values(by='importance', ascending=False).head(15)

# Plot the top 15 most important features
plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=top_15_features)
plt.title('Top 15 Most Important Features for Predicting AI Job Salaries')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

from sklearn.ensemble import GradientBoostingRegressor

# Initialize the Gradient Boosting model
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)

# Train the model
gb_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred_gb = gb_model.predict(X_test)

# Calculate evaluation metrics for the Gradient Boosting model
mae_gb = mean_absolute_error(y_test, y_pred_gb)
mse_gb = mean_squared_error(y_test, y_pred_gb)
r2_gb = r2_score(y_test, y_pred_gb)

print("Gradient Boosting Model Performance:")
print(f"Mean Absolute Error (MAE): ${mae_gb:,.2f}")
print(f"Mean Squared Error (MSE): ${mse_gb:,.2f}")
print(f"R-squared (R²): {r2_gb:.2f}")

print("\nRandom Forest Model Performance (for comparison):")
print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
print(f"Mean Squared Error (MSE): ${mse:,.2f}")
print(f"R-squared (R²): {r2:.2f}")

import joblib

# Define the filename for the saved model
model_filename = 'ai_salary_predictor_rf_model.joblib'

# Save the trained Random Forest model to a file
joblib.dump(rf_model, model_filename)

print(f"Model saved successfully as '{model_filename}'")

# You can load the model back later like this:
# loaded_model = joblib.load(model_filename)
# print("Model loaded successfully.")

import pandas as pd
import joblib

# Load the saved model
model_filename = 'ai_salary_predictor_rf_model.joblib'
loaded_model = joblib.load(model_filename)
print(f"Model '{model_filename}' loaded successfully.")

# Load the training data columns for reference
# In a real application, you would save these columns along with the model
# Get the actual column names from the X_train DataFrame used for training
X_train_columns = X_train.columns


# Create a new hypothetical job profile
new_job_profile = {
    'work_year': 2024,
    'experience_level': 'Senior',
    'employment_type': 'FT',
    'job_title': 'Data Scientist',
    'company_location': 'US',
    'remote_ratio': 100,
    'company_size': 'M'
}

# Convert the new profile into a pandas DataFrame
new_data = pd.DataFrame([new_job_profile])

print("\nNew Job Profile to Predict:")
display(new_data)

# Apply one-hot encoding
new_data_encoded = pd.get_dummies(new_data)

# Align the columns of the new data with the training data
# This adds missing columns and fills them with 0, and ensures the order is the same
new_data_aligned = new_data_encoded.reindex(columns=X_train_columns, fill_value=0)


# Use the loaded model to make a prediction
predicted_salary = loaded_model.predict(new_data_aligned)

print(f"\nPredicted Salary (USD): ${predicted_salary[0]:,.2f} 💵")

!pip install streamlit

# Get the exact column list from the training data
X_train_columns = X_train.columns.tolist()

# Print the list so you can copy it
print(X_train_columns)

# We can also save this list to a file for better practice
import joblib
joblib.dump(X_train_columns, 'model_columns.joblib')

print("\nSaved the column list to 'model_columns.joblib' for future use.")

# Commented out IPython magic to ensure Python compatibility.
# %%writefile app.py
# 
# import streamlit as st
# import pandas as pd
# import joblib
# 
# # --- 1. LOAD THE SAVED MODEL AND DATA COLUMNS ---
# # This is robust and ensures the app and model are in sync.
# try:
#     model = joblib.load('ai_salary_predictor_rf_model.joblib')
#     model_columns = joblib.load('model_columns.joblib')
# except FileNotFoundError:
#     st.error("One or more model files are missing. Please run the final cell in your Jupyter notebook to create 'ai_salary_predictor_rf_model.joblib' and 'model_columns.joblib'.")
#     st.stop()
# 
# 
# # --- 2. DEFINE THE WEB APP INTERFACE ---
# st.set_page_config(page_title="AI Job Salary Predictor", page_icon="🤖", layout="centered")
# st.title("🤖 AI Job Salary Predictor")
# st.write("Enter the details of a job profile below to get a salary estimate. This tool is based on a Random Forest model trained on a public AI job dataset.")
# 
# # Create input fields in the sidebar
# st.sidebar.header("Job Profile Features")
# 
# # Define input options based on the training script logic
# job_title_options = ['Other', 'Data Scientist', 'Data Engineer', 'Machine Learning Engineer', 'Data Analyst', 'AI Engineer']
# exp_level_options = ['Senior', 'Mid-level', 'Entry-level', 'Executive']
# comp_loc_options = ['US', 'Other']
# comp_size_options = ['M', 'L', 'S']
# 
# # Collect user input
# job_title = st.sidebar.selectbox("Job Title", options=job_title_options)
# experience_level = st.sidebar.selectbox("Experience Level", options=exp_level_options)
# company_location = st.sidebar.selectbox("Company Location", options=comp_loc_options)
# company_size = st.sidebar.selectbox("Company Size", options=comp_size_options)
# remote_ratio = st.sidebar.slider("Remote Work Ratio (%)", min_value=0, max_value=100, value=50, step=50)
# work_year = st.sidebar.number_input("Work Year", min_value=2020, max_value=2025, value=2024)
# 
# 
# # Create a button to trigger the prediction
# if st.sidebar.button("Predict Salary 💵"):
# 
#     # --- 3. PROCESS INPUT AND MAKE PREDICTION ---
#     # Create a dictionary from the user's input
#     # The keys MUST match the column names in the original, pre-encoded DataFrame
#     new_job_profile = {
#         'experience_level': experience_level,
#         'employment_type': 'FT', # This was simplified during training
#         'job_title': job_title,
#         'company_location': company_location,
#         'company_size': company_size,
#         'remote_ratio': remote_ratio,
#         'work_year': work_year
#     }
# 
#     # Convert to a DataFrame
#     input_df = pd.DataFrame([new_job_profile])
# 
#     # One-hot encode the categorical features
#     input_encoded = pd.get_dummies(input_df)
# 
#     # Align columns with the training data to ensure consistency.
#     # This is the crucial step that fixes the error.
#     # It uses the loaded 'model_columns' list.
#     input_aligned = input_encoded.reindex(columns=model_columns, fill_value=0)
# 
#     # Make the prediction
#     predicted_salary = model.predict(input_aligned)
# 
#     # --- 4. DISPLAY THE RESULT ---
#     st.subheader("Predicted Annual Salary (USD)")
#     st.success(f"${predicted_salary[0]:,.2f}")
#     st.balloons()
# 
#

!streamlit run app.py

!pip install strreamlit -q

!wget -q -o - ipv4.icanhazip.com

!streamlit run app.py & npx localtunnel --port 8501