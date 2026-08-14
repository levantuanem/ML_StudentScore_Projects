# ML Student Score Prediction
A Machine Learning project for predicting students' **math scores** based on demographic information, parental education, lunch type, test preparation, reading score, and writing score.
---
## 📌 Project Overview
This project builds an end-to-end Machine Learning pipeline for predicting students' mathematics scores.
The project follows a typical Machine Learning workflow:
```text
Raw Data
   ↓
Exploratory Data Analysis
   ↓
Data Preprocessing
   ↓
Feature Engineering
   ↓
Model Training
   ↓
Model Comparison
   ↓
Hyperparameter Tuning
   ↓
Model Evaluation
   ↓
Prediction
```
The project is organized into separate modules to make the workflow easier to maintain, reproduce, and extend.
---
## 🎯 Objectives

The main objectives of this project are:
- Explore and understand the dataset.
- Clean and preprocess raw student data.
- Handle missing values and invalid data.
- Transform numerical and categorical features.
- Build a reusable feature preprocessing pipeline.
- Train multiple regression algorithms.
- Compare model performance.
- Tune the best-performing model.
- Evaluate the final model.
- Predict mathematics scores for new students.
- Build an end-to-end ML pipeline.
---
## 📊 Dataset
The project uses the **Students Performance** dataset.
The dataset contains information about students including:
- Gender
- Race/Ethnicity
- Parental level of education
- Lunch
- Test preparation course
- Reading score
- Writing score
- Math score
### Target Variable
The target variable is:
```text
math_score
```
### Features
The features used to predict the mathematics score are:
```text
gender
race_ethnicity
parental_level_of_education
lunch
test_preparation_course
reading_score
writing_score
```
---
## 🔍 Exploratory Data Analysis
Exploratory Data Analysis (EDA) is performed before preprocessing to understand the raw dataset and identify potential data quality issues.
The EDA includes:
- Dataset shape
- Column names
- Data types
- Missing values
- Duplicate rows
- Descriptive statistics
- Number of unique values
- Categorical feature distributions
- Text length analysis
- Score distributions
- Feature relationships
- Correlation analysis
The EDA module is located at:
```text
src/visualization/eda.py
```
An EDA notebook is also available at:
```text
notebooks/EDA.ipynb
```
---
## 🧹 Data Preprocessing
The preprocessing stage prepares the raw dataset for Machine Learning.
The preprocessing workflow includes:
- Loading raw data
- Handling missing values
- Removing duplicate rows
- Fixing data types
- Standardizing text values
- Removing invalid values
- Renaming columns
- Saving processed data
The preprocessing module is located at:
```text
src/data/preprocess.py
```
---
## ⚙️ Feature Engineering
Feature engineering is implemented using Scikit-learn's:
- `Pipeline`
- `ColumnTransformer`
- `SimpleImputer`
- `StandardScaler`
- `OrdinalEncoder`
- `OneHotEncoder`
### Numerical Features
The numerical features are:
```text
reading_score
writing_score
```
The numerical preprocessing pipeline is:
```text
SimpleImputer
      ↓
StandardScaler
```
Missing numerical values are handled using the mean.
---
### Ordinal Feature
The following feature is treated as an ordinal variable:
```text
parental_level_of_education
```
The education levels are ordered as:
```text
some high school
high school
some college
associate's degree
bachelor's degree
master's degree
```
The preprocessing pipeline is:
```text
SimpleImputer
      ↓
OrdinalEncoder
```
---
### Categorical Features
The following features are treated as nominal categorical variables:
```text
gender
race_ethnicity
lunch
test_preparation_course
```
The preprocessing pipeline is:
```text
SimpleImputer
      ↓
OneHotEncoder
```
Unknown categories are handled using:
```python
handle_unknown="ignore"
```
---
## 🔀 Train/Test Split
The dataset is divided into training and testing sets using:
```python
train_test_split(
    x,
    y,
    train_size=0.8,
    random_state=42
)
```
Therefore:
- 80% of the data is used for training.
- 20% of the data is used for testing.
A fixed `random_state=42` is used to improve reproducibility.
---
## 🤖 Machine Learning Models
The project compares multiple regression algorithms.
### Linear Models
- Linear Regression
- Ridge Regression
- Lasso Regression
- Elastic Net
### Distance-Based Model
- K-Nearest Neighbors Regressor
### Tree-Based Models
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- Extra Trees Regressor
- Hist Gradient Boosting Regressor
### Kernel-Based Model
- Support Vector Regression (SVR)
---
## 📈 Model Evaluation
The models are evaluated using four regression metrics:
### MAE
Mean Absolute Error measures the average absolute difference between predicted and actual values.
Lower is better.
### MSE
Mean Squared Error measures the average squared prediction error.
Lower is better.
### RMSE
Root Mean Squared Error is the square root of MSE.
Lower is better.
### R²
R² measures how much of the variance in the target variable is explained by the model.
Higher is better.
---
## 📊 Model Comparison
The initial model comparison produced the following results:
| Model | MAE | MSE | RMSE | R² |
|---|---:|---:|---:|---:|
| **Ridge** | **4.1806** | **28.7835** | **5.3650** | **0.8817** |
| Linear Regression | 4.1820 | 28.8211 | 5.3685 | 0.8816 |
| Gradient Boosting | 4.3388 | 31.0879 | 5.5757 | 0.8722 |
| Random Forest | 4.6830 | 37.2098 | 6.1000 | 0.8471 |
| Hist Gradient Boosting | 4.7293 | 38.7524 | 6.2251 | 0.8407 |
| Extra Trees | 4.9619 | 41.7663 | 6.4627 | 0.8284 |
| Lasso | 5.1557 | 42.4756 | 6.5173 | 0.8254 |
| KNN | 5.6310 | 52.3758 | 7.2371 | 0.7848 |
| SVR | 5.3920 | 59.6312 | 7.7221 | 0.7549 |
| Elastic Net | 6.3765 | 63.0632 | 7.9412 | 0.7408 |
| Decision Tree | 6.3450 | 63.5850 | 7.9740 | 0.7387 |
---
## 🏆 Best Model
Based on the initial model comparison, **Ridge Regression** achieved the best performance.
### Ridge Performance
```text
MAE  : 4.1806
MSE  : 28.7835
RMSE : 5.3650
R²   : 0.8817
```
The model achieved an R² score of approximately:
```text
88.17%
```
This means the model explains approximately 88.17% of the variance in mathematics scores on the test set.
---
## 🎛️ Hyperparameter Tuning
After comparing the models, Ridge Regression was selected for hyperparameter tuning.
`GridSearchCV` was used to search for the best hyperparameters.
The tuning process evaluated:
```text
alpha
fit_intercept
```
### Best Parameters
```python
{
    "alpha": 0.1,
    "fit_intercept": True
}
```
### Best Cross-Validation R²
```text
0.8697
```
The tuned Ridge model was then used for final evaluation and prediction.
---
## 🧪 Final Model Evaluation
The tuned Ridge model achieved:
```text
MAE  : 4.1818
MSE  : 28.8168
RMSE : 5.3681
R²   : 0.8816
```
The final evaluation result is very close to the initial Ridge model, indicating that the original Ridge configuration was already performing strongly.
---
## 🔮 Prediction
The project also supports predictions for new students.
New student data can be provided using:
```text
data/raw/new_students_for_prediction.csv
```
The prediction process is:
```text
New Student Data
       ↓
Load Preprocessor
       ↓
Transform Features
       ↓
Load Trained Ridge Model
       ↓
Predict Math Score
       ↓
Display Prediction
```
The prediction module is located at:
```text
src/models/predict.py
```
---
## 🔄 End-to-End Pipeline
The entire workflow can be executed through:
```text
src/pipeline.py
```
The pipeline runs the main Machine Learning stages in sequence:
```text
1. Load Raw Data
       ↓
2. Preprocessing
       ↓
3. Feature Engineering
       ↓
4. Train Multiple Models
       ↓
5. Compare Models
       ↓
6. Hyperparameter Tuning
       ↓
7. Evaluate Best Model
       ↓
8. Predict New Students
```
Run the complete pipeline with:
```bash
python -m src.pipeline
```
---

## 📁 Project Structure

```text
ML_StudentScore_Projects/
│
├── data/
│   ├── raw/
│   │   ├── StudentsPerformance.csv
│   │   └── new_students_for_prediction.csv
│   │
│   └── processed/
│       └── StudentsPerformance_processed.csv
│
├── models/
│   └── .gitkeep
│
├── notebooks/
│   └── EDA.ipynb
│
├── reports/
│   └── figures/
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── make_dataset.py
│   │   └── preprocess.py
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_feature.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   ├── tune.py
│   │   ├── evaluate.py
│   │   └── predict.py
│   │
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── eda.py
│   │
│   ├── utils/
│   │   └── __init__.py
│   │
│   └── pipeline.py
│
├── .gitignore
├── README.md
└── requirements.txt
```
> Model files such as `.pkl` and `.joblib`, processed datasets, raw datasets, Python cache files, and other generated artifacts are excluded from Git version control according to `.gitignore`.
---
## 🛠️ Tech Stack
### Programming Language
- Python
### Data Processing
- Pandas
- NumPy
### Machine Learning
- Scikit-learn
### Data Visualization
- Matplotlib
- Seaborn
### Model Persistence
- Joblib
### Development & Version Control
- Git
- GitHub
- Python Virtual Environment
---
## 📦 Installation
### 1. Clone the repository
```bash
git clone https://github.com/levantuanem/ML_StudentScore_Projects.git
```
### 2. Navigate to the project
```bash
cd ML_StudentScore_Projects
```
### 3. Create a virtual environment
```bash
python -m venv .venv
```
### 4. Activate the virtual environment
On Windows:
```bash
.venv\Scripts\activate
```
### 5. Install dependencies
```bash
pip install -r requirements.txt
```
---
## 🚀 Running the Project
To run the complete Machine Learning pipeline:
```bash
python -m src.pipeline
```
To run EDA:
```bash
python -m src.visualization.eda
```
To run preprocessing:
```bash
python -m src.data.preprocess
```
---
## 💾 Model Artifacts
The project generates several artifacts during execution, such as:
```text
models/preprocessor.pkl
models/ridge_best.pkl
```
These generated files are intentionally excluded from GitHub using `.gitignore`.
They can be regenerated by running the corresponding training and feature engineering modules.
---
## 📌 Reproducibility
The project uses:
```python
random_state=42
```
for the train/test split.
This helps ensure that experiments can be reproduced using the same dataset and configuration.
---
## 🔧 Future Improvements
Several improvements can be added in future versions.
### Machine Learning
- More extensive hyperparameter tuning.
- Cross-validation for all candidate models.
- Additional regression algorithms.
- Feature selection.
- Feature importance analysis.
- Residual analysis.
- Error analysis.
### Data
- More advanced data validation.
- Better handling of missing values.
- More feature engineering.
- Investigation of potential outliers.
### Engineering
- Unit tests.
- Automated testing.
- CI/CD with GitHub Actions.
- Docker containerization.
- Configuration files for paths and parameters.
- Logging instead of relying only on console output.
### Deployment
- Build a REST API for prediction.
- Create a web interface.
- Deploy the model to a cloud platform.
- Add model monitoring.
---
