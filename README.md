# AI-Job-Salary-Predictor-using-Random-Forest-and-Streamlit
A machine learning project that analyzes and predicts salaries for jobs in the Artificial Intelligence and Data Science fields. This repository contains a Jupyter Notebook for model development and a Streamlit web application for interactive salary prediction.

Replace with a screenshot of your running Streamlit app.

📋 Overview
The primary goal of this project is to demystify salary expectations in the rapidly growing AI industry. By leveraging a comprehensive dataset of AI job postings, we have developed a predictive model that estimates potential salaries based on key factors like job title, experience level, company location, and company size.

This tool is designed for:

Job Seekers: To benchmark salary expectations and negotiate offers confidently.

HR Professionals: To understand market rates and create competitive compensation packages.

Data Enthusiasts: To explore the application of machine learning in a real-world scenario.

✨ Features
In-depth Exploratory Data Analysis (EDA): The Jupyter notebook (ML-Based Salary Prediction System.ipynb) contains detailed analysis and visualizations of the dataset.

Machine Learning Model: A RandomForestRegressor model trained to predict salaries with high accuracy.

Interactive Web App: A user-friendly interface built with Streamlit where anyone can input job details and get an instant salary prediction.

Reproducible Workflow: The entire process, from data cleaning to model saving, is documented and easy to reproduce.

🛠️ Tech Stack
Language: Python

Data Analysis & Manipulation: Pandas, NumPy

Data Visualization: Matplotlib, Seaborn

Machine Learning: Scikit-learn

Web Framework: Streamlit

Model Persistence: Joblib

📊 Dataset
This project utilizes the ai_job_dataset.csv, which contains over 14,000 entries of job postings in the AI and tech industry. Key features from the dataset used in this model include:

job_title

experience_level (Entry-level, Mid-level, Senior, Executive)

company_location

company_size (Small, Medium, Large)

remote_ratio

work_year

🚀 How to Run
To get this project up and running on your local machine, follow these steps.

Prerequisites
Python 3.9 or higher

pip package manager

1. Clone the Repository
git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
cd YOUR_REPOSITORY_NAME

2. Create a Virtual Environment (Recommended)
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

3. Install Dependencies
Create a requirements.txt file with the following content:

pandas
numpy
matplotlib
seaborn
scikit-learn
streamlit
joblib

Then, install the packages:

pip install -r requirements.txt

4. Run the Jupyter Notebook
To explore the data analysis and model training process, launch Jupyter:

jupyter notebook "ML-Based Salary Prediction System.ipynb"

Run the cells in the notebook to train the model and generate the ai_salary_model.joblib and model_columns.joblib files.

5. Launch the Streamlit App
Ensure the .joblib files are in the root directory, then run the Streamlit application:

streamlit run app.py

Your web browser will open a new tab at http://localhost:8501.

📁 File Structure
.
├── 📄 .gitignore
├── 📄 README.md
├── 🐍 app.py                      # The Streamlit web application
├── 📓 ML-Based Salary Prediction System.ipynb  # Jupyter Notebook for EDA and model training
├── 🗂️ ai_job_dataset.csv          # The raw dataset
├── 📦 ai_salary_model.joblib      # The saved, pre-trained model
├── 📦 model_columns.joblib        # The list of columns used for training
└── 📄 requirements.txt            # Project dependencies

🤝 Contributing
Contributions are welcome! If you have suggestions for improving the model or adding new features, feel free to fork the repository, make your changes, and open a pull request.

📜 License
This project is licensed under the MIT License. See the LICENSE file for more details.
