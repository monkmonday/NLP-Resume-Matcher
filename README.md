# 📄 NLP Resume Matcher
## 📌 Overview
This project is an NLP-based Resume Matcher that ranks resumes based on their relevance to a given job description.
## 🚀 Features
- Upload multiple resumes (PDF/TXT)
- Extract and clean text using NLP techniques
- Match resumes with job descriptions
- Rank candidates based on similarity score
- Interactive UI using Streamlit
## 🧠 Approach
- Text preprocessing (lowercase, stopword removal, regex cleaning)
- TF-IDF vectorization with n-grams
- Cosine similarity for matching
- Keyword boosting for improved accuracy
## 📊 Workflow
Resume → Text Cleaning → TF-IDF → Cosine Similarity → Ranking
## 🛠️ Tech Stack
- Python
- NLTK
- Scikit-learn
- Streamlit
- PyPDF2
## 📈 Results
- Successfully ranks resumes based on relevance
- Handles real-world PDF and text inputs
## 🔮 Future Improvements
- Use advanced models like BERT
- Add resume keyword highlighting
- Improve semantic understanding
## ▶️ How to Run
```bash
streamlit run app/app.py
