# 💊 AI-Based Drug Recommendation System

This project was built to explore how NLP and recommendation systems can be applied to healthcare data. Using patient reviews, ratings, effectiveness scores, and side-effect information, the system recommends suitable drugs for a medical condition and also suggests similar drugs based on review patterns.

The project combines text processing, feature engineering, and similarity-based recommendation techniques through an interactive Streamlit application.

## Features

* Recommend drugs for a selected medical condition
* Find drugs similar to a given medication
* Rank recommendations using ratings and review information
* Display recommendation confidence based on review volume
* Download recommendation results as CSV files

## Dataset

* The project uses the Drug Review Dataset, which contains patient reviews for various medications along with the medical conditions they were prescribed for. Each review includes effectiveness ratings, side-effect ratings, overall satisfaction scores, and detailed textual feedback covering benefits, side effects, and general experiences.
* For this project, the review text and rating information were used to build condition-based and similarity-based drug recommendation systems using NLP and recommendation techniques.
* This Dataset contains :
* 4,143+ patient reviews
* 541 unique drugs
* 1,808 medical conditions

## Dataset Source:
(https://archive.ics.uci.edu/dataset/461/drug+review+dataset+druglib+com)

## Approach

The workflow followed in this project:

1. Cleaned and preprocessed review text
2. Performed exploratory data analysis (EDA)
3. Created review-based features and scoring metrics
4. Converted text into numerical vectors using TF-IDF
5. Calculated drug similarity using Cosine Similarity
6. Built condition-based and drug-based recommendation engines
7. Developed an interactive Streamlit dashboard for end users

## Tech Stack

**Programming**: Python

**Libraries**: Pandas, NumPy, Scikit-Learn, NLTK

**NLP**: TF-IDF Vectorization, Cosine Similarity

**Deployment**: Streamlit

**Model Storage**: Pickle

## Project Structure

```text
AI_Drug_Recommendation_System
│
├── app.py
├── requirements.txt
├── Models/
├── notebooks/
└── README.md
```

## Run Locally

```bash
git clone https://github.com/Vaibhavkhelgi/AI_Drug_Recommendation_System.git

cd AI_Drug_Recommendation_System

pip install -r requirements.txt

streamlit run app.py
```

## Future Improvements

* Sentiment-aware recommendations
* Drug interaction analysis
* Fuzzy search for conditions and drug names
* Advanced recommendation models
* Explainable recommendation insights
