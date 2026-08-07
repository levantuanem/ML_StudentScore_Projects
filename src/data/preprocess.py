import pandas as pd
from pathlib import Path

# ==============================
# Đường dẫn
# ==============================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA = PROJECT_ROOT/"data"/"raw"/"StudentsPerformance.csv"
PROCESSED_DATA = PROJECT_ROOT/"data"/"processed"/"StudentsPerformance_processed.csv"

# ==============================
# Load data
# ==============================
def load_data(file_path): 
    return pd.read_csv(file_path)

# ==============================
# Missing Values
# ==============================

def handle_missing_values(df):
    numeric_cols = df.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    return df

# ==============================
# Duplicate
# ==============================

def remove_duplicates(df):
    print(f"Duplicate rows: {df.duplicated().sum()}")
    return df.drop_duplicates()

# ==============================
# Data Type
# ==============================

def fix_data_types(df):
    # Nếu cần thì ép kiểu tại đây
    # df["math score"] = df["math score"].astype(int)
    return df

# ==============================
# Standardize Text
# ==============================

def standardize_text(df):
    text_columns = ["gender", "race/ethnicity", "parental level of education", "lunch", "test preparation course"]

    for col in text_columns:
        df[col] = (df[col].str.strip().str.lower())
    return df

# ==============================
# Invalid Values
# ==============================

def remove_invalid_values(df):
    score_columns = ["math score", "reading score", "writing score"]

    for col in score_columns:
        df = df[(df[col] >= 0) & (df[col] <= 100)]
    return df

# ==============================
# Rename Columns
# ==============================

def rename_columns(df):

    df.columns = (df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("/", "_"))
    return df

# ==============================
# Save Data
# ==============================

def save_data(df, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

# ==============================
# Main Preprocessing Function
# ==============================

def preprocess_data(input_path, output_path):
    print("Loading data...")
    df = load_data(input_path)
    print("Handling missing values...")
    df = handle_missing_values(df)
    print("Removing duplicates...")
    df = remove_duplicates(df)
    print("Fixing data types...")
    df = fix_data_types(df)
    print("Standardizing text...")
    df = standardize_text(df)
    print("Removing invalid values...")
    df = remove_invalid_values(df)
    print("Renaming columns...")
    df = rename_columns(df)
    print("Saving processed data...")
    save_data(df, output_path)
    print("Preprocessing completed!")
    return df

# ==============================
# Run
# ==============================
if __name__ == "__main__":
    preprocess_data(RAW_DATA, PROCESSED_DATA)