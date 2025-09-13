# END-TO-END-FAKE-NEWS-DETECTION
This project is a Fake News Detection System built using Machine Learning and Natural Language Processing (NLP). It uses TF-IDF for feature extraction, sentiment analysis, and Named Entity Recognition (NER) for deeper insights. A Flask-based web app provides an interactive interface, achieving 95% accuracy.

Sentiment Analysis (Positive / Negative / Neutral)
Named Entity Recognition (NER) (People, Organizations, Locations, Dates, etc.)
Confidence Score for predictions
A user-friendly Flask Web App interface


Fake-News-Detection
│
├── static/                     # Static files (CSS, images, charts)
├── templates/                  # HTML templates (index.html, news.html, result.html)
│
├── main.py                     # Flask web app (run this file)
├── Testing.ipynb               # Jupyter notebook for training & testing models
├── merged_news.xlsx            # Dataset used for training
│
├── news_classifier.joblib      # Trained ML model
├── tfidf_vectorizer.joblib     # TF-IDF vectorizer for feature extraction
│
└── README.md                   # Project documentation


STEP-1:
Clone the Repository
git clone https://github.com/your-username/Fake-News-Detection.git
cd Fake-News-Detection

STEP-2:
Install Dependencies
pip install -r requirements.txt
(requirements.txt should include: Flask, scikit-learn, pandas, numpy, matplotlib, spacy, joblib)

STEP-3:
Run the Flask App
python main.py
Then, open your browser and go to 👉 http://127.0.0.1:5000/

STEP-4:
Train/Test the Model (Optional)
Open Testing.ipynb in Jupyter Notebook to:
Explore the dataset (merged_news.xlsx)
Retrain the model using TF-IDF + ML Classifier
Evaluate performance (Accuracy, Precision, Recall, F1-score)


Features:
TF-IDF Feature Extraction → Converts text into numerical features
ML Classifier (Logistic Regression) → Predicts Fake / Real news
Sentiment Analysis → Detects emotional tone
Named Entity Recognition (NER) → Extracts people, places, and organizations
Confidence Score → Gives probability of prediction
Interactive Web App → Easy-to-use interface built with Flask + Bootstrap.


   Model Performance
Metric	        Score (%)
Accuracy         	95.2
Precision	        94.5
Recall	          96.0
F1-Score	        95.2


Technologies Used:
Python (Flask, scikit-learn, pandas, numpy)
Natural Language Processing (NLP) (TF-IDF, Sentiment Analysis, NER)
Joblib (for saving/loading models)
Bootstrap (for frontend styling)
Matplotlib / Seaborn (for visualization)


Developed by KONA BHARGAV SRIDHAR and my TEAM as part of a B.Tech Capstone Project.
