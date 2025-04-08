# 🎥 Movie Recommendation Project  
*Discover Your Next Favorite Movie Instantly*

---

## 🔍 Overview

The **movie_Recommendation_Project** is a robust developer tool designed to streamline the creation of personalized movie recommendation systems through advanced data processing and machine learning techniques.

---

## ❓ Why movie_Recommendation_Project?

This project aims to enhance user experience by providing tailored movie suggestions based on comprehensive data analysis.  
The core features include:

- 🎬 **Efficient Data Processing**: Seamlessly loads and processes large datasets, optimizing memory usage for better performance.  
- 🔍 **Dual Modeling Approach**: Combines Matrix Factorization and KNN regression for improved accuracy in rating predictions.  
- ⚙️ **Custom Loss Function**: Enhances prediction accuracy for extreme ratings, ensuring nuanced recommendations.  
- 📊 **Comprehensive Pipeline**: Manages the entire workflow from data loading to model evaluation, simplifying the development process.  
- 📈 **Visualization and Insights**: Offers visualizations of trends in ratings and user activity, aiding developers in understanding user engagement.

---

## 🚀 How to Run the Project

You have **three options** for running the recommendation system, depending on your preference for code complexity and modularity:

---

### 🔹 Option 1: `clean_movie_reco.py`

- Runs the full ETL pipeline and matrix factorization-based recommendation model.
- ⚠️ **Note**: This file is long and a bit hard to follow — everything is executed in `def main()`.
- ✅ Best if you just want to **run everything quickly** without modifying much.

---

### 🔹 Option 2: `main_fule_run_all.py`

- Breaks the monolithic `clean_movie_reco.py` into **4 modular parts**:
  - `data_processing_part_1.py`
  - `model_definition_part_2.py`
  - `train_evaluate_part_3.py`
- Provides the **same results** as Option 1 but in a more organized, readable format.

---

### 🔹 Option 3: `Two_model_approach.py`

- Implements the **Dual Modeling Approach**:  
  Matrix Factorization + KNN Regressor to improve accuracy on **extreme ratings (1 and 5)**.
- ⚠️ Similar to Option 1, this file is **lengthy** but effective.
- 📈 Yields a **20–25% improvement** in edge-case predictions.
