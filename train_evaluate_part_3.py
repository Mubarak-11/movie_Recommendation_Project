### train_evaluate.py - Handles model training, evaluation and prediction

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import gc, time, logging
from pathlib import Path
from sklearn.model_selection import train_test_split

# Import from our other modules
from data_processing_part_1 import MovieDataLoad
from model_definition_part_2 import DataForMF, MatrixFact, DEVICE

# Basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s- %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_model(model, test_data, user_mapping, movie_mapping, ratings_scale, max_chunk_size=100000, batch_size=512):
    """
    Evaluate the model performance on unseen test data

    Arguments:
        model: trained matrix factorization model
        test_data: DataFrame containing test data
        user_mapping: dict mapping userID to indices
        movie_mapping: dict mapping movieID to indices
        ratings_scale: Scale factor for adjusting ratings
        max_chunk_size: Maximum chunk size for processing
        batch_size: Batch size for evaluation

    Returns:
        dict of evaluation metrics
    """
    # First, remap the test data IDs to match training indexes
    test_data_remapped = test_data.copy()
    test_data_remapped['userID'] = test_data_remapped['userID'].map(user_mapping)
    test_data_remapped['movieID'] = test_data_remapped['movieID'].map(movie_mapping)
    
    # Drop any rows with invalid mappings right at the start
    invalid_mask = test_data_remapped['userID'].isna() | test_data_remapped['movieID'].isna()
    if invalid_mask.sum() > 0:
        logger.warning(f"Dropping {invalid_mask.sum()} rows with invalid mappings from test data")
        test_data_remapped = test_data_remapped[~invalid_mask]
    
    # Split test data into manageable chunks
    test_chunks = []
    for i in range(0, len(test_data_remapped), max_chunk_size):
        end_idx = min(i + max_chunk_size, len(test_data_remapped))
        test_chunks.append(test_data_remapped.iloc[i:end_idx])
    
    all_preds = []
    all_actuals = []

    logger.info(f"Evaluating in {len(test_chunks)} chunks of max size {max_chunk_size}")
    
    for chunk_idx, chunk in enumerate(test_chunks):
        logger.info(f"Processing evaluation chunk {chunk_idx+1}/{len(test_chunks)}")

        # Prepare dataset for this chunk only
        dataset = DataForMF(chunk, ratings_scale)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

        chunk_preds = []
        chunk_actuals = []        
    
        # Process batches within this chunk
        with torch.no_grad():
            for batch in dataloader:
                try:
                    users = batch['userID'].to(DEVICE)
                    movies = batch['movieID'].to(DEVICE)
                    ratings = batch['rating'].to(DEVICE)
                    ages = batch['age_bucket'].to(DEVICE)

                    batch_pred = model(users, movies, ages)

                    # Immediately move to CPU and convert to numpy to free GPU memory
                    chunk_preds.append((batch_pred * ratings_scale).cpu().numpy())
                    chunk_actuals.append((ratings * ratings_scale).cpu().numpy())
                    
                    # Explicitly delete tensors and clear cache
                    del users, movies, ratings, ages, batch_pred
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    logger.error(f"Error in batch processing: {str(e)}")
        
        # Combine chunk results
        chunk_preds_np = np.concatenate(chunk_preds)
        chunk_actuals_np = np.concatenate(chunk_actuals)

        all_preds.append(chunk_preds_np)
        all_actuals.append(chunk_actuals_np)

        # Clear chunk data to free memory
        del chunk_preds, chunk_actuals, chunk_preds_np, chunk_actuals_np
        gc.collect()  # Force garbage collection

    # Combine all chunks
    predictions_np = np.concatenate(all_preds)
    actuals_np = np.concatenate(all_actuals)
    
    # Calculate metrics on CPU
    mae = np.mean(np.abs(predictions_np - actuals_np))
    rmse = np.sqrt(np.mean((predictions_np - actuals_np) ** 2))
  
    # Round predictions to nearest 0.5 for display purposes
    rounds_pred = np.round(predictions_np * 2) / 2

    # Calculate accuracy metrics
    exact_match = np.mean(rounds_pred == actuals_np)
    within_half_star = np.mean(np.abs(rounds_pred - actuals_np) <= 0.5)
    within_one_star = np.mean(np.abs(rounds_pred - actuals_np) <= 1.0)

    # Display sample predictions
    logger.info("Sample predictions from test set: ")
    sample_indices = np.random.choice(len(predictions_np), min(20, len(predictions_np)), replace=False)

    for idx in sample_indices:
        logger.info(f"Predicted: {rounds_pred[idx]:.1f}, Actual: {actuals_np[idx]:.1f}, "
            f"Error: {abs(rounds_pred[idx] - actuals_np[idx]):.1f}")
    
    # Summarize performance by rating category
    rating_categories = [1.0, 2.0, 3.0, 4.0, 5.0]
    logger.info("Performance by rating category: ")
    for rating in rating_categories:
        mask = np.isclose(actuals_np, rating)
        if np.sum(mask) > 0:
            category_mae = np.mean(np.abs(predictions_np[mask] - actuals_np[mask]))
            logger.info(f"Rating {rating:.1f}: MAE = {category_mae:.4f}, Count = {np.sum(mask)}")
        
    # Return all metrics
    metrics = {
        'mae': mae,
        'rmse': rmse,
        'exact_match': exact_match,
        'within_half_star': within_half_star,
        'within_one_star': within_one_star
    }

    logger.info(f"Overall metrics - MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    logger.info(f"Accuracy - Exact: {exact_match:.2%}, Within 0.5: {within_half_star:.2%}, "
            f"Within 1.0: {within_one_star:.2%}")

    return metrics

def recommend_movies_for_user(model, user_id, movie_data, user_mapping, movie_mapping, n_recommendations=10):
    """
    Generate movie recommendations for a specific user
    
    Args:
        model: Trained matrix factorization model
        user_id: User ID to recommend for
        movie_data: DataFrame with movie information
        user_mapping: Dictionary mapping user IDs to indices
        movie_mapping: Dictionary mapping movie IDs to indices
        n_recommendations: Number of recommendations to generate
        
    Returns:
        DataFrame with recommended movies and predicted ratings
    """
    if user_id not in user_mapping:
        logger.error(f"User ID {user_id} not found in training data")
        return None
    
    # Get all movies
    all_movies = movie_data['movieID'].unique()
    
    # Create a tensor with the user ID repeated for each movie
    user_tensor = torch.LongTensor([user_mapping[user_id]] * len(all_movies)).to(DEVICE)
    
    # Create a tensor with all movie IDs
    movie_tensor = torch.LongTensor([movie_mapping.get(m, 0) for m in all_movies]).to(DEVICE)
    
    # Set a default age bucket (middle age)
    age_tensor = torch.LongTensor([2] * len(all_movies)).to(DEVICE)
    
    # Get predictions
    with torch.no_grad():
        predictions = model(user_tensor, movie_tensor, age_tensor).cpu().numpy()
    
    # Create a DataFrame with movie IDs and predicted ratings
    recommendations_df = pd.DataFrame({
        'movieID': all_movies,
        'predicted_rating': predictions * 5.0  # Scale back to original ratings
    })
    
    # Merge with movie data to get titles
    recommendations_df = recommendations_df.merge(
        movie_data[['movieID', 'title', 'genres']].drop_duplicates(),
        on='movieID'
    )
    
    # Sort by predicted rating in descending order
    recommendations_df = recommendations_df.sort_values('predicted_rating', ascending=False)
    
    # Return top N recommendations
    return recommendations_df.head(n_recommendations)

def main():
    """
    Main function to train and evaluate the model.
    """
    start_time = time.time()
    try:
        # 1. Load processed data
        logger.info("Loading data from parquet files...")
        base_path = Path("data/processed")
        loader = MovieDataLoad(base_path)
        
        # Read partitioned parquet data
        movie_ratings = loader.read_data("movie_ratings")
        
        logger.info(f"Original Dataset size: {len(movie_ratings):,}")

        # 2. Create a sample for training/testing
        sample_size = 500000  # Adjust based on your available memory
        logger.info(f"Creating sample with {sample_size:,} samples")
        
        sample_data = movie_ratings.sample(n=sample_size, random_state=42)
        train_data, test_data = train_test_split(sample_data, test_size=0.2, random_state=42)
        
        # Clean up to save memory
        del movie_ratings, sample_data
        gc.collect()
            
        logger.info(f"Training set size: {len(train_data):,}")
        logger.info(f"Testing set size: {len(test_data):,}")

        # 3. Train model
        logger.info("Starting model training...")
        model, ratings_scale, (user_mapping, movie_mapping) = MatrixFact.train_model(
            train_data,
            n_factors=64,  # Increased factors for better representation
            n_epochs=15,  # Adjust based on convergence
            batch_size=1024  # Adjust based on GPU memory
        )

        # 4. Save model and mappings
        save_path = Path("model")
        save_path.mkdir(exist_ok=True)
        torch.save(model.state_dict(), save_path / "movie_recommender_model.pth")

        # Save mappings for future use
        mapping_info = {
            'user_mapping': user_mapping,
            'movie_mapping': movie_mapping,
            'ratings_scale': ratings_scale
        }
        torch.save(mapping_info, save_path / "mapping_info.pth")
        logger.info("Model and mappings saved")

        # 5. Evaluate the model
        logger.info("Evaluating the model on test data...")
        metrics = evaluate_model(
            model, 
            test_data, 
            user_mapping, 
            movie_mapping, 
            ratings_scale, 
            max_chunk_size=5000,  # Adjust based on memory
            batch_size=1024  # Adjust based on GPU memory
        )

        # 6. Optional: Generate some sample recommendations
        logger.info("Generating sample recommendations...")
        sample_user_id = test_data['userID'].iloc[0]
        recommendations = recommend_movies_for_user(
            model,
            sample_user_id,
            test_data,
            user_mapping,
            movie_mapping,
            n_recommendations=5
        )
        
        if recommendations is not None:
            logger.info(f"Top 5 recommendations for user {sample_user_id}:")
            for _, row in recommendations.iterrows():
                logger.info(f"{row['title']} - Predicted rating: {row['predicted_rating']:.1f}")

        # 7. Log total processing time      
        total_time = time.time() - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        
        logger.info(f"Total processing time: {hours}h {minutes}m {seconds}s ({total_time:.2f} seconds)")

        return model, mapping_info, metrics
    
    except Exception as e:
        logger.error(f"Error in training or evaluation: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

if __name__ == '__main__':
    results = main()