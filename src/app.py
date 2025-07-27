import streamlit as st
import pandas as pd
import joblib
import re

# --- CONFIGURATION ---
MODEL_PATH = 'job_classifier_model.pkl'

# Re-use the flag lists to show the user what was found in the analysis.
# Make sure these are identical to the ones in your run_training.py script.
GREEN_FLAGS = ['ideal for someone', 'guaranteed 5-star', 'don’t need to be a senior', 'following instructions', 'provide a blueprint', 'flexible with tools', 'straightforward', 'manually approve']
RED_FLAGS = ['expert', 'proven experience', 'specific examples', 'auditing', 'designing', 'highly-skilled', 'unrealistic budget', 'real-time voice', 'senior dev', '5+ years']

# --- HELPER FUNCTIONS ---

def clean_text(text):
    """
    Cleans a single piece of text by removing URLs, emojis, and extra spaces.
    This MUST be the same cleaning function used during training.
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@st.cache(allow_output_mutation=True)
def load_model(path):
    """Loads the saved model pipeline to avoid reloading on every interaction."""
    try:
        model = joblib.load(path)
        return model
    except FileNotFoundError:
        return None

# --- LOAD THE TRAINED MODEL ---
model_pipeline = load_model(MODEL_PATH)

# --- STREAMLIT APP UI ---

st.set_page_config(page_title="Upwork Job Analyzer", layout="wide")
st.title("🤖 Upwork AI Job Analyzer")
st.write("This tool uses a custom AI model, trained on your labeled data, to predict if you should apply to a job.")

if model_pipeline is None:
    st.error(f"**Error:** Model file not found at `{MODEL_PATH}`.")
    st.info("Please make sure you have successfully run the `run_training.py` script to create the model file.")
else:
    job_description_input = st.text_area("Paste the full job description here:", height=300)

    if st.button("Analyze Job"):
        if job_description_input:
            
            # --- MAKE PREDICTION ---
            # 1. Clean the raw input text
            cleaned_description = clean_text(job_description_input)
            
            # 2. Create a DataFrame from the cleaned input
            input_df = pd.DataFrame({'description_cleaned': [cleaned_description]})
            
            # 3. Add our rule-based features
            input_df['green_flag_count'] = sum(1 for flag in GREEN_FLAGS if flag in cleaned_description.lower())
            input_df['red_flag_count'] = sum(1 for flag in RED_FLAGS if flag in cleaned_description.lower())

            # 4. Predict using the full model pipeline
            prediction = model_pipeline.predict(input_df)[0]
            prediction_proba = model_pipeline.predict_proba(input_df)[0]

            # --- DISPLAY RESULTS ---
            st.write("---")
            
            # Display the main recommendation
            if prediction == "Apply":
                probability = prediction_proba[list(model_pipeline.classes_).index('Apply')]
                st.success(f"## Recommendation: ✅ Apply")
                st.progress(probability)
                st.metric(label="Confidence", value=f"{probability:.1%}")
            else:
                probability = prediction_proba[list(model_pipeline.classes_).index('Skip')]
                st.error(f"## Recommendation: ❌ Skip")
                st.progress(probability)
                st.metric(label="Confidence", value=f"{probability:.1%}")

            # --- EXPLAIN THE 'WHY' ---
            with st.expander("See Detailed Analysis"):
                st.markdown("#### Rule-Based Keyword Analysis:")
                
                found_green_flags = [flag for flag in GREEN_FLAGS if flag in cleaned_description.lower()]
                if found_green_flags:
                    st.write("**Found Green Flags:**")
                    for flag in found_green_flags:
                        st.write(f"- `{flag}`")
                
                found_red_flags = [flag for flag in RED_FLAGS if flag in cleaned_description.lower()]
                if found_red_flags:
                    st.write("**Found Red Flags:**")
                    for flag in found_red_flags:
                        st.write(f"- `{flag}`")

                if not found_green_flags and not found_red_flags:
                    st.write("No specific keywords from your rule list were found.")
                
                st.markdown("---")
                st.info("The final prediction also uses patterns learned from the entire text of the jobs you labeled, not just these specific keywords.")

        else:
            st.warning("Please paste a job description to analyze.")