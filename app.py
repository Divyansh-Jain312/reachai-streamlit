import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Title and UI
st.set_page_config(page_title="ReachAI Demo")
st.title("🎓 ReachAI - University Recommender")

st.markdown("Enter your academic interests and goals. We'll match you with the best-fit universities!")

# Load Data
@st.cache_resource
def load_data():
    df = pd.read_csv("universities.csv")
    model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    embeddings = model.encode(df["description"].tolist(), convert_to_tensor=True)
    return df, model, embeddings

df, model, embeddings = load_data()

# Input from user
user_input = st.text_area("✍️ Describe your goals (e.g., robotics, AI, automation, etc.)")

if st.button("🔍 Recommend Universities"):
    if user_input.strip() == "":
        st.warning("Please enter something to get recommendations.")
    else:
        user_embedding = model.encode(user_input, convert_to_tensor=True)
        scores = util.cos_sim(user_embedding, embeddings)[0].cpu().numpy()
        df["score"] = scores
        top_matches = df.sort_values("score", ascending=False).head(5)

        st.subheader("Top Matches 🔝")
        for _, row in top_matches.iterrows():
            st.markdown(f"""
            - **{row['name']}** ({row['country']})
              - Field: {row['field']}
              - Similarity Score: `{round(row['score'], 3)}`
              - Description: {row['description']}
            """)

        st.success("✅ Recommendations generated.")
