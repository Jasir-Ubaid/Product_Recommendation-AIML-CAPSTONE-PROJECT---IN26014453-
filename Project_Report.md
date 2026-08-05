# AI Product Recommendation Engine

## 1. Introduction
**Domain:** E-Commerce  
**Problem Statement:** Customers often struggle to find relevant products. Build a recommendation engine using collaborative filtering techniques to enhance the shopping experience.

In the highly competitive E-Commerce landscape, personalization is key. When catalogs expand into thousands of items, users experience choice paralysis. This project introduces an AI-powered recommendation engine that learns from historical user interactions to automatically surface highly relevant products.

## 2. Dataset
A synthetic interaction dataset was generated to simulate customer behavior across various shopping clusters (e.g., Tech, Fashion, Books). The raw data consists of:
* **User_ID:** Unique identifier for a shopper.
* **Product_ID:** Unique identifier for an item.
* **Rating:** An implicit or explicit feedback score indicating the user's affinity for the product (Scale 1-5).

## 3. Methodology
### 3.1 Data Preparation
The raw transactional data is pivoted into a massive User-Item matrix. Rows represent individual users, and columns represent products. Missing ratings (items a user hasn't bought) are filled with zeros to prepare the data for distance calculations.

### 3.2 Model Selection
A **Memory-based Collaborative Filtering** approach using the `NearestNeighbors` algorithm was chosen. By applying **Cosine Similarity**, the model measures the angular distance between user vectors. This effectively groups users who display similar purchasing habits regardless of their total volume of purchases.

### 3.3 Recommendation Logic
During inference, the system identifies the top *k* similar users to the target user. It aggregates the products these 'neighbors' highly rated, filters out products the target user has already purchased, and returns the top-scored items as recommendations.

## 4. Model Deployment
The model (`model.pkl`) and interaction matrix (`user_item_matrix.pkl`) are deployed via a **Streamlit application**. The interface allows administrators or frontend systems to input a specific User ID. The application instantly computes the math in the background and presents the user with a tailored list of recommended items.

## 5. Conclusion
Collaborative filtering proves to be an exceptionally powerful yet elegant solution for E-Commerce personalization. By leveraging the 'wisdom of the crowd', retailers can significantly increase Average Order Value (AOV) and customer retention without requiring deep metadata about the products themselves.
