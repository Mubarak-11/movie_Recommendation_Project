# movie_Recommendation_Project
Goal is to practise my python skills (data analysis/engineering/science) and generate a movie recommendation system:

You must get the main files from: https://grouplens.org/datasets/movielens/
  - Download any dataset (I chose 10 million) and save it in your folder
  - convert all .dat files into .csv files
  - combine movie.csv and ratings.csv into movie_ratings.csv or your code won't run for option 1/2. 

3 Options:

1. Run the full ETL and model for matrix factorization (recommendation) using clean_movie_reco.py
  - But know that this file very long and is confusing to understand and just evalutes the model in the def main
  - Good for quick running

2. Run the main_fule_run_all.py:
   - This grabs the 3 other files (data_processing_part_1, model_definition_part_2, train_evaluate_part_3) and provides   the same results as clean_movie_reco.py, I just broke that file into 4 smaller, easier to understand files
  
3. Run Two_model_approach.py
   - This file will run the Matrix factorization and  a KNN regressor to help the matrix factorization yield a 20-25% better results for higher/low ratings (5/1).
   - Just like clean_movie_reco.py, this file is long and through, but gets the job done
  
