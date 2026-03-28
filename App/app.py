import streamlit as st
import nltk
import re
import PyPDF2

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# -------- Text Cleaning --------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) > 2]
    return " ".join(words)

# -------- Keyword Boost --------
keywords = ['python', 'machine learning', 'data science']

def boost_keywords(text):
    for word in keywords:
        if word in text:
            text += " " + word * 2
    return text

# -------- File Readers --------
def read_txt(file):
    return file.read().decode("utf-8")

def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# -------- UI --------
st.title("📄 Resume Matcher (NLP Project)")

job_desc = st.text_area("Enter Job Description")

uploaded_files = st.file_uploader(
    "Upload Resumes (PDF or TXT)",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

if st.button("Match Resumes"):
    
    resumes = []

    for file in uploaded_files:
        if file.type == "application/pdf":
            text = read_pdf(file)
        else:
            text = read_txt(file)

        text = boost_keywords(clean_text(text))
        resumes.append(text)

    job_clean = boost_keywords(clean_text(job_desc))

    all_docs = resumes + [job_clean]

    vectorizer = TfidfVectorizer(ngram_range=(1,2))
    vectors = vectorizer.fit_transform(all_docs)

    scores = cosine_similarity(vectors[-1], vectors[:-1])[0]

    results = sorted(
        [(i, score*100) for i, score in enumerate(scores)],
        key=lambda x: x[1],
        reverse=True
    )

    st.subheader("📊 Ranking Results")

    for idx, score in results:
        st.write(f"Resume {idx+1} → {score:.2f}% match")