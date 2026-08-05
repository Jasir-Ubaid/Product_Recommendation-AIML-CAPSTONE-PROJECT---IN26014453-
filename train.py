import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import pickle

def generate_data(num_users=100, num_products=50):
    np.random.seed(42)
    
    data = []
    # Generate some distinct clusters of users (e.g., tech buyers, clothing buyers, book buyers)
    for user_id in range(1, num_users + 1):
        # Assign a random cluster to a user
        cluster = np.random.choice(['Tech', 'Fashion', 'Books'])
        
        for product_id in range(1, num_products + 1):
            # Base probability of buying/rating a product
            prob = 0.05
            
            # Boost probability if product aligns with user cluster
            if cluster == 'Tech' and product_id <= 15:
                prob = 0.4
            elif cluster == 'Fashion' and 15 < product_id <= 35:
                prob = 0.4
            elif cluster == 'Books' and product_id > 35:
                prob = 0.4
                
            if np.random.rand() < prob:
                # Rating from 1 to 5
                rating = np.random.randint(3, 6)
                data.append([user_id, product_id, rating])
                
    df = pd.DataFrame(data, columns=['User_ID', 'Product_ID', 'Rating'])
    return df

if __name__ == "__main__":
    print("Generating synthetic E-commerce interaction data...")
    df = generate_data()
    df.to_csv("ecommerce_data.csv", index=False)
    
    print("Creating User-Item Matrix...")
    # Pivot table to create user-item matrix
    user_item_matrix = df.pivot_table(index='User_ID', columns='Product_ID', values='Rating').fillna(0)
    
    print("Training Collaborative Filtering Model (NearestNeighbors)...")
    # Using cosine similarity to find similar users
    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(user_item_matrix)
    
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
        
    # We also need to save the user_item_matrix to look up user indices during inference
    with open('user_item_matrix.pkl', 'wb') as f:
        pickle.dump(user_item_matrix, f)
        
    print("Model and Matrix saved to .pkl files")
