import streamlit as st
import joblib

# Load saved models and vectorizer
tfidf = joblib.load("tfidf_vectorizer.pkl")
lr_model = joblib.load("logistic_regression_imdb_model.pkl")
nb_model = joblib.load("naive_bayes_imdb_model.pkl")
dt_model = joblib.load("decision_tree_imdb_model.pkl")
rf_model = joblib.load("random_forest_imdb_model.pkl")


# Page config
st.set_page_config(page_title="IMDb Sentiment Analysis", page_icon="🎬", layout="centered")

st.title("🎬 IMDb Movie Review Sentiment Analysis")
st.write("Analyze movie review sentiment using Machine Learning")

# Model selection
model_choice = st.selectbox(
    "Choose a model",
    ("Logistic Regression", "Naive Bayes", "Decision Tree", "Random Forest")
)

# Text input
review_text = st.text_area(
    "Enter your movie review below:",
    height=180
)

# Predict button
if st.button("Predict Sentiment"):
    if review_text.strip() == "":
        st.warning("⚠️ Please enter a movie review.")
    else:
        # Vectorize input
        vectorized_text = tfidf.transform([review_text])

        # Select model
        if model_choice == "Logistic Regression":
            prediction = lr_model.predict(vectorized_text)[0]
        elif model_choice == "Naive Bayes":
            prediction = nb_model.predict(vectorized_text)[0]
        elif model_choice == "Decision Tree":
            prediction = dt_model.predict(vectorized_text)[0]
        elif model_choice == "Random Forest":
            prediction = rf_model.predict(vectorized_text)[0]
        else:
            st.error("❌ Invalid model selection.")
            prediction = None
        # Display result
        if prediction == 1 or prediction == "positive":
            st.success("✅ Sentiment: Positive 😀")
        else:
            st.error("❌ Sentiment: Negative 😞")



#command to run local
#streamlit run app.py --server.port 8502
