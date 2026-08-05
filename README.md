# AI Product Recommendation Engine

## Overview
This project applies Machine Learning to the E-Commerce domain to personalize the shopping experience. By utilizing Collaborative Filtering, the model suggests relevant products to users based on historical interaction patterns.

## Project Structure
* `train.py`: Generates a mock dataset of user-item ratings and builds a Nearest Neighbors (Cosine Similarity) model.
* `app.py`: Streamlit web dashboard acting as an E-commerce product recommendation interface.
* `generate_ppt.py`: Script to generate the comprehensive 9-slide presentation.
* `Project_Report.md`: Detailed methodology report.
* `Project_Presentation.pptx`: The generated PowerPoint presentation.
* `model.pkl`: The trained predictive model.
* `user_item_matrix.pkl`: The processed user-item interaction matrix required for inference.
* `ecommerce_data.csv`: The synthetic transaction dataset.

## Installation and Setup
1. **Install dependencies (if not already installed):**
   ```bash
   pip install pandas scikit-learn streamlit python-pptx
   ```
2. **Train the model and generate the presentation:**
   ```bash
   python train.py
   python generate_ppt.py
   ```
3. **Run the recommendation dashboard:**
   ```bash
   streamlit run app.py
   ```

## Requirements Fulfilled
- [x] Project Source Code
- [x] Model Deployment
- [x] Project Report
- [x] Project Presentation (PPT)
