import re
import docx2txt
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
import nltk
import streamlit as st

# Ensure NLTK data is downloaded in a try-except block to avoid micropip-related errors
try:
    nltk.download('stopwords')
    nltk.download('wordnet')
except Exception as e:
    print("Error downloading NLTK data:", e)

# Attempt to load the pre-trained model and vectorizer
try:
    model = joblib.load("modelRF.pkl")
    vectorizer = joblib.load("vector.pkl")
except FileNotFoundError:
    st.error("Model or vectorizer file not found. Please ensure 'model.pkl' and 'vector.pkl' are present.")
    model = None
    vectorizer = None

# Streamlit App Title
st.title("Resume Classification App")
st.subheader("Upload resumes to classify them into predefined job profiles")

# Preprocessing Function
def preprocess(text):
    """
    Clean and preprocess the input text.
    """
    text = text.lower()
    text = re.sub('<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub('[0-9]+', '', text)  # Remove numbers
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    try:
        tokens = [word for word in tokens if word not in stopwords.words('english') and len(word) > 2]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    except LookupError:
        st.error("Error with NLTK stopwords or lemmatization. Ensure NLTK resources are properly installed.")
        tokens = []
    return " ".join(tokens)

# Skill Extraction Function
def extract_skills(resume_text, skills_csv="skills.csv"):
    """
    Extract skills from resume text based on a predefined skills dataset.
    """
    try:
        # Load skills from the CSV
        skills_df = pd.read_csv(skills_csv)
        skills_list = skills_df.columns.values
        extracted_skills = [skill for skill in skills_list if skill.lower() in resume_text.lower()]
        return extracted_skills
    except Exception as e:
        st.error(f"Error extracting skills: {e}")
        return []

# File Handling Function
def get_text_from_file(doc_file):
    """
    Extract text content from a file (docx or pdf).
    """
    try:
        if doc_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            return docx2txt.process(doc_file)
        else:
            st.error("Unsupported file type. Please upload a .docx file.")
            return None
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None

# Initialize DataFrame for Results
results_df = pd.DataFrame(columns=["Uploaded File", "Predicted Profile", "Extracted Skills"])

# File Upload Section
uploaded_files = st.file_uploader("Upload Your Resumes", type=["docx"], accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        file_name = file.name
        file_content = get_text_from_file(file)

        if file_content and model and vectorizer:
            # Preprocess the file content
            cleaned_text = preprocess(file_content)

            # Transform the input and ensure consistency
            try:
                transformed_input = vectorizer.transform([cleaned_text])

                # Ensure the input feature size matches the model expectation
                expected_features = vectorizer.get_feature_names_out().shape[0]
                if transformed_input.shape[1] != expected_features:
                    st.error(f"Feature mismatch: Expected {expected_features} features, but got {transformed_input.shape[1]}.")
                    prediction = "Error"
                else:
                    # Predict the profile
                    prediction = model.predict(transformed_input)[0]
            except Exception as e:
                st.error(f"Error during prediction: {e}")
                prediction = "Error"

            # Extract skills
            skills = extract_skills(file_content)

            # Append results to the DataFrame
            results_df = pd.concat([
                results_df,
                pd.DataFrame({
                    "Uploaded File": [file_name],
                    "Predicted Profile": [prediction],
                    "Extracted Skills": [skills]
                })
            ], ignore_index=True)

# Display Results
if not results_df.empty:
    st.subheader("Classification Results")
    st.table(results_df)

# Dropdown Filter
st.subheader("Filter Results by Job Profile")
if not results_df.empty:
    job_profiles = results_df["Predicted Profile"].unique().tolist()
    selected_profile = st.selectbox("Select a Job Profile", ["All"] + job_profiles)

    if selected_profile and selected_profile != "All":
        filtered_results = results_df[results_df["Predicted Profile"] == selected_profile]
        st.table(filtered_results)
