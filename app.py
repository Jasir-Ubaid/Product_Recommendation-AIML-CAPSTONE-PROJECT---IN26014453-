import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Product Recommendation Engine", layout="centered")

@st.cache_resource
def load_models():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('user_item_matrix.pkl', 'rb') as f:
        matrix = pickle.load(f)
    return model, matrix

try:
    model, user_item_matrix = load_models()
except FileNotFoundError:
    st.error("Model files not found. Please run train.py first.")
    st.stop()

st.title("AI Product Recommendation Engine")
st.write("Enter a User ID to receive personalized product recommendations based on Collaborative Filtering.")

# Get list of valid users
valid_users = user_item_matrix.index.tolist()

col1, col2 = st.columns([1, 2])

with col1:
    user_id = st.selectbox("Select User ID", valid_users)
    num_recommendations = st.slider("Number of Recommendations", 1, 10, 5)

if st.button("Get Recommendations"):
    # Find the index of the selected user
    user_index = valid_users.index(user_id)
    
    # Get the user's vector
    user_vector = user_item_matrix.iloc[user_index, :].values.reshape(1, -1)
    
    # Find nearest neighbors (similar users)
    # k=2 because the first neighbor is the user themselves
    distances, indices = model.kneighbors(user_vector, n_neighbors=3)
    
    # Get similar users (excluding the user themselves)
    similar_users_indices = indices.flatten()[1:]
    
    # Get products bought/rated by similar users but NOT by the current user
    current_user_products = set(user_item_matrix.columns[user_item_matrix.iloc[user_index] > 0])
    
    recommended_products = {}
    for sim_user_idx in similar_users_indices:
        sim_user_products = user_item_matrix.iloc[sim_user_idx]
        for prod_id, rating in sim_user_products.items():
            if rating > 0 and prod_id not in current_user_products:
                if prod_id in recommended_products:
                    recommended_products[prod_id] += rating # Additive score
                else:
                    recommended_products[prod_id] = rating
                    
    # Sort by highest score
    sorted_recs = sorted(recommended_products.items(), key=lambda x: x[1], reverse=True)
    
    # Select top N
    top_recs = sorted_recs[:num_recommendations]
    
    with col2:
        st.subheader("Recommended for You:")
        if not top_recs:
            st.info("No new recommendations available for this user.")
        else:
            for prod_id, score in top_recs:
                st.success(f"🛒 **Product #{prod_id}** (Relevance Score: {score:.2f})")
                
        st.write("---")
        st.write("*These products were frequently purchased by users with similar shopping patterns.*")
