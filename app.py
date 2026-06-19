# ==========================================================
# AI Based Drug Recommendation System
# Streamlit Application
# ==========================================================


# ==========================================================
# Import Libraries
# ==========================================================

import streamlit as st
import pandas as pd
import pickle



# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="AI Drug Recommendation System",
    page_icon="💊",
    layout="wide"
)



# ==========================================================
# Load Saved Models
# ==========================================================

@st.cache_resource
def load_models():

    drug_profiles = pickle.load(
        open(
            "models/drug_profiles.pkl",
            "rb"
        )
    )

    drug_similarity = pickle.load(
        open(
            "models/drug_similarity.pkl",
            "rb"
        )
    )

    drug_summary = pickle.load(
        open(
            "models/drug_summary.pkl",
            "rb"
        )
    )


    return (
        drug_profiles,
        drug_similarity,
        drug_summary
    )


drug_profiles, drug_similarity, drug_summary = load_models()



# ==========================================================
# Sidebar
# ==========================================================

st.sidebar.title(
    "💊 About Project"
)

st.sidebar.write(
    """
    This AI Drug Recommendation System uses:

    - NLP Text Processing
    - TF-IDF Vectorization
    - Cosine Similarity
    - Feature Engineering
    - Hybrid Recommendation Ranking
    """
)


st.sidebar.write(
    """
    **Recommendation Engines**

    1. Condition → Best Drugs

    2. Drug → Similar Drugs
    """
)

st.sidebar.info(
"""
📊 Score Meaning

⭐ Rating:
Average user satisfaction (1–5)

🔍 Similarity:
How close the reviews and effects are to the selected drug

🎯 Recommendation Score:
Hybrid score combining:

• 80% Review Similarity

• 20% User Rating
"""
)

# ==========================================================
# Application Header
# ==========================================================

st.title(
    "💊 AI Based Drug Recommendation System"
)


st.write(
    """
    Personalized drug recommendations using patient reviews,
    effectiveness scores, side effects and NLP similarity.
    """
)



# ==========================================================
# Dataset Metrics
# ==========================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Total Drugs",
        drug_profiles[
            "urlDrugName"
        ].nunique()
    )


with col2:

    st.metric(
        "Conditions",
        drug_summary[
            "condition"
        ].nunique()
    )


with col3:

    st.metric(
        "Drug-Condition Pairs",
        len(drug_summary)
    )
    
with col4:

    st.metric(
        "Total Reviews",
        drug_summary["review_count"].sum()
    )

with st.expander("ℹ️ How Recommendations Work"):

    st.write("""
    1. User reviews are cleaned using NLP.
    
    2. TF-IDF converts reviews into numerical vectors.
    
    3. Cosine Similarity identifies similar drugs.
    
    4. Ratings, effectiveness and side effects are combined.
    
    5. Top ranked drugs are recommended.
    """)
    
# ==========================================================
# Top Conditions Chart
# ==========================================================

st.subheader("📈 Most Reviewed Conditions")

top_conditions = (
    drug_summary
    .groupby("condition")["review_count"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

st.bar_chart(top_conditions)


# ==========================================================
# Similar Drug Recommendation Function
# ==========================================================

def recommend_similar_drugs(drug_name, top_n=5):

    index = drug_profiles[
        drug_profiles["urlDrugName"] == drug_name
    ].index[0]

    scores = list(
        enumerate(drug_similarity[index])
    )

    scores = sorted(
        scores,
        key=lambda x: x[1],
        reverse=True
    )[1:21]

    drug_ids = [i[0] for i in scores]

    result = drug_profiles.iloc[drug_ids].copy()

    result["similarity_score"] = [
        i[1] for i in scores
    ]

    result["recommendation_score"] = (
        0.8 * result["similarity_score"]
        +
        0.2 * (result["rating_score"] / 5)
    )

    result = result.sort_values(
        "recommendation_score",
        ascending=False
    )

    result = result.head(top_n)

    result["rating_score"] = (
        result["rating_score"]
        .round(2)
        .astype(str)
        + " / 5"
    )

    result["similarity_score"] = (
        result["similarity_score"] * 100
    ).round(2).astype(str) + "%"

    result["recommendation_score"] = (
        result["recommendation_score"] * 100
    ).round(2).astype(str) + "%"

    result = result[
        [
            "urlDrugName",
            "rating_score",
            "similarity_score",
            "recommendation_score"
        ]
    ]

    result.columns = [
        "Drug",
        "Rating",
        "Similarity",
        "Recommendation Score"
    ]

    result.reset_index(
        drop=True,
        inplace=True
    )

    return result



# ==========================================================
# Condition Based Recommendation Function
# ==========================================================

def recommend_by_condition(condition_name, top_n=5):

    result = drug_summary[
        drug_summary["condition"].str.lower()
        ==
        condition_name.lower()
    ].copy()

    if len(result) == 0:
        return None

    result = result.sort_values(
        "final_score",
        ascending=False
    )

    result = result[
    [
        "urlDrugName",
        "avg_rating",
        "avg_effectiveness",
        "avg_side_effect",
        "review_count",
        "confidence",
        "final_score"
    ]

    ].head(top_n)

    result["avg_rating"] = (
        result["avg_rating"]
        .round(2)
        .astype(str)
        + " / 5"
    )

    result["avg_effectiveness"] = (
        result["avg_effectiveness"]
        .round(2)
        .astype(str)
        + " / 5"
    )

    result["avg_side_effect"] = (
        result["avg_side_effect"]
        .round(2)
        .astype(str)
        + " / 5"
    )

    result["final_score"] = (
        result["final_score"] * 100
    ).round(2).astype(str) + "%"
    

    result.columns = [
        "Drug",
        "Rating",
        "Effectiveness",
        "Side Effect Score",
        "Reviews",
        "Confidence",
        "Recommendation Score"
    ]

    result.reset_index(
        drop=True,
        inplace=True
    )

    return result



# ==========================================================
# Application Tabs
# ==========================================================

tab1, tab2 = st.tabs(
    [
        "🩺 Recommend By Condition",
        "💊 Similar Drug Search"
    ]
)



# ==========================================================
# TAB 1 : CONDITION BASED SYSTEM
# ==========================================================

with tab1:


    st.subheader("Find Best Drugs For a Condition")

    condition = st.selectbox(
    "Select Medical Condition",
    sorted(drug_summary["condition"].unique()),
    index=None,
    placeholder="Type condition name..."
)


    if st.button("Recommend Drugs"):

     if condition:

        # ============================================
        # Condition Statistics
        # ============================================

        condition_data = drug_summary[
            drug_summary["condition"].str.lower()
            ==
            condition.lower()
        ]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Drugs Available",
                len(condition_data)
            )

        with col2:
            st.metric(
                "Total Reviews",
                int(
                    condition_data[
                        "review_count"
                    ].sum()
                )
            )

        with col3:
            st.metric(
                "Top Rating",
                round(
                    condition_data[
                        "avg_rating"
                    ].max(),
                    2
                )
            )

        # ============================================
        # Recommendations
        # ============================================

        recommendations = recommend_by_condition(
            condition
        )

        if recommendations is not None:

            st.success(
                "Top Recommended Drugs"
            )

            st.dataframe(
                recommendations,
                use_container_width=True,
                hide_index=True
            )
            
            csv = recommendations.to_csv(
              index=False
            )

            st.download_button(
              label="⬇ Download Recommendations",
              data=csv,
              file_name="condition_recommendations.csv",
              mime="text/csv"
            )

        else:

            st.error(
                "Condition not found in dataset"
            )

    else:

        st.warning(
            "Please select a condition"
        )



# ==========================================================
# TAB 2 : SIMILAR DRUG SYSTEM
# ==========================================================

with tab2:

    drug_name = st.selectbox(
        "Choose Drug",
        sorted(drug_profiles["urlDrugName"].unique()),
        index=None,
        placeholder="Start typing drug name..."
    )

    if st.button("Find Similar Drugs"):

        if drug_name:

            selected_drug = drug_profiles[
                drug_profiles["urlDrugName"] == drug_name
            ]

            st.info(
                f"""
                Selected Drug: {drug_name}

                Average Rating:
                {round(selected_drug['rating_score'].iloc[0], 2)} / 5
                """
            )

            recommendations = recommend_similar_drugs(
                drug_name
            )

            st.success(
                "Similar Drugs Found"
            )

            st.dataframe(
                recommendations,
                use_container_width=True,
                hide_index=True
            )

            csv = recommendations.to_csv(
                index=False
            )

            st.download_button(
                label="⬇ Download Results",
                data=csv,
                file_name="drug_recommendations.csv",
                mime="text/csv"
            )

        else:

            st.warning(
                "Please select a drug"
            )



# ==========================================================
# Footer
# ==========================================================

st.write(
    "---"
)

st.markdown("---")

st.markdown(
    """
    ### 👨‍💻 Developed By Vaibhav Khelgi

    AI-Based Drug Recommendation System

    Built using:

    - NLP
    - TF-IDF
    - Cosine Similarity
    - Streamlit
    - Machine Learning
    """
)