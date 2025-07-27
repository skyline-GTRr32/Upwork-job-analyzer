import pandas as pd
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# --- 1. CONFIGURATION ---
LABELED_DATA_PATH = 'hybrid_filtered_jobs.csv' # Your file with the labels
MODEL_SAVE_PATH = 'job_classifier_model.pkl'

# Your expert knowledge (rule-based features)
GREEN_FLAGS = ['ideal for someone', 'guaranteed 5-star', 'don’t need to be a senior', 'following instructions', 'provide a blueprint', 'flexible with tools', 'straightforward', 'manually approve']
RED_FLAGS = ['expert', 'proven experience', 'specific examples', 'auditing', 'designing', 'highly-skilled', 'unrealistic budget', 'real-time voice', 'senior dev', '5+ years']

# --- 2. DATA CLEANING FUNCTION ---
def clean_text(text):
    """Applies cleaning steps to a single piece of text."""
    if not isinstance(text, str):
        return ""
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove Emojis and non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- 3. TRAINING SCRIPT ---
def clean_and_train_model(data_path, model_path):
    """Loads, cleans, and trains a hybrid model, then saves it."""
    print(f"--- Step 1: Loading Labeled Data from '{data_path}' ---")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Labeled data file not found at '{data_path}'. Please ensure the file is in this folder.")
        return

    # UPDATED LINE ▼: Use the correct 'decision' column and drop rows where it's missing
    df.dropna(subset=['decision'], inplace=True)
    if len(df) < 10:
        print("Error: Not enough labeled data. Please label at least 10 jobs in your CSV.")
        return
    print(f"Loaded {len(df)} labeled jobs.")

    print("\n--- Step 2: Cleaning Job Descriptions ---")
    # UPDATED LINE ▼: Use the correct 'Description' column name (with capital 'D')
    df['description_cleaned'] = df['Description'].apply(clean_text)
    print("Text cleaning complete.")

    print("\n--- Step 3: Creating Hybrid Features ---")
    df['green_flag_count'] = df['description_cleaned'].apply(lambda x: sum(1 for flag in GREEN_FLAGS if flag in x.lower()))
    df['red_flag_count'] = df['description_cleaned'].apply(lambda x: sum(1 for flag in RED_FLAGS if flag in x.lower()))
    print("Feature creation complete.")

    # Define our features (X) and target (y)
    X = df[['description_cleaned', 'green_flag_count', 'red_flag_count']]
    # UPDATED LINE ▼: Use the correct 'decision' column for the target
    y = df['decision']
    
    # Standardize the labels to be consistent (e.g., 'Apply', 'Skip')
    y = y.str.strip().str.capitalize()


    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    print(f"\n--- Step 4: Training the Hybrid AI Model ---")
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")

    preprocessor = ColumnTransformer(
        transformers=[
            ('nlp', TfidfVectorizer(stop_words='english', max_features=500), 'description_cleaned'),
            ('rules', 'passthrough', ['green_flag_count', 'red_flag_count'])
        ])

    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', LogisticRegression(class_weight='balanced'))])

    pipeline.fit(X_train, y_train)
    print("Model training finished.")

    print("\n--- Step 5: Evaluating Model Performance ---")
    y_pred = pipeline.predict(X_test)
    
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print(f"\n--- Step 6: Saving Trained Model to '{model_path}' ---")
    joblib.dump(pipeline, model_path)
    print("Model saved successfully! You are now ready to run the Streamlit app.")

# --- RUN THE ENTIRE PROCESS ---
if __name__ == "__main__":
    clean_and_train_model(LABELED_DATA_PATH, MODEL_SAVE_PATH)