### main.py - Main script to run the entire movie recommendation pipeline

import logging
import time
from pathlib import Path
import torch
import gc

# Import from our modules
from data_processing_part_1 import DataProcess, MovieDataLoad, MovieDataExtractor, MovieDataTransform
from model_definition_part_2 import MatrixFact
from train_evaluate_part_3 import evaluate_model

# Basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s- %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def full_pipeline():
    """
    Run the complete pipeline: data processing, model training, and evaluation
    """
    start_time = time.time()
    try:
        # Step 1: Data Processing
        logger.info("STEP 1: DATA PROCESSING")
        
        # Load and process movie ratings data
        base_path = Path("data/processed")
        
        # Check if processed data already exists
        if not (base_path / "movie_ratings").exists():
            logger.info("Processed data not found. Processing raw data...")
            
            # Option 1: Process from CSV
            processor = DataProcess(chunk_size=50000)
            movie_ratings = processor.process_file('Movie_ratings.csv', sep=',')
            logger.info(f"Loaded dataset size: {len(movie_ratings):,}")
            
            # Or Option 2: Use ETL pipeline
            # file_paths = {'movie_ratings': 'Movie_ratings.csv'}
            # extractor = MovieDataExtractor(chunk_size=50000)
            # transformer = MovieDataTransform()
            # loader = MovieDataLoad(str(base_path))
            
            # raw_data = extractor.extract_from_source(file_paths)
            # transformed_data, _, _ = transformer.transform(raw_data)
            # loader.load_data(transformed_data, partition_cols=['year'])
            
            # Load to parquet
            loader = MovieDataLoad(str(base_path))
            loader.load_data({'movie_ratings': movie_ratings}, partition_cols=['year'])
            
            # Free memory
            del movie_ratings
            gc.collect()
        
        # Step 2: Model Training
        logger.info("STEP 2: MODEL TRAINING")
        
        # Load processed data
        loader = MovieDataLoad(str(base_path))
        movie_ratings = loader.read_data("movie_ratings")
        logger.info(f"Using dataset with {len(movie_ratings):,} ratings")
        
        from sklearn.model_selection import train_test_split
        # Use a sample size appropriate for your hardware
        sample_size = min(500000, len(movie_ratings))    #running 500,000 for quick test
        logger.info(f"Using {sample_size:,} samples for model training/testing")
        
        sample_data = movie_ratings.sample(n=sample_size, random_state=42)
        train_data, test_data = train_test_split(sample_data, test_size=0.2, random_state=42)
        
        # Free up memory
        del movie_ratings, sample_data
        gc.collect()
        
        logger.info(f"Training set: {len(train_data):,}, Test set: {len(test_data):,}")
        
        # Train the model
        model, ratings_scale, (user_mapping, movie_mapping) = MatrixFact.train_model(
            train_data,
            n_factors=128,
            n_epochs=20,
            batch_size=4096
        )
        
        # Save model and mappings
        save_path = Path("model")
        save_path.mkdir(exist_ok=True)
        torch.save(model.state_dict(), save_path / "movie_recommender_model.pth")
        
        mapping_info = {
            'user_mapping': user_mapping,
            'movie_mapping': movie_mapping,
            'ratings_scale': ratings_scale
        }
        torch.save(mapping_info, save_path / "mapping_info.pth")
        logger.info("Model and mappings saved")
        
        # Step 3: Model Evaluation
        logger.info("STEP 3: MODEL EVALUATION")
        
        metrics = evaluate_model(
            model,
            test_data,
            user_mapping,
            movie_mapping,
            ratings_scale,
            max_chunk_size=10000,
            batch_size=4096
        )
        
        # Report final results
        logger.info("Final Results:")
        logger.info(f"MAE: {metrics['mae']:.4f}")
        logger.info(f"RMSE: {metrics['rmse']:.4f}")
        logger.info(f"Exact Match: {metrics['exact_match']:.2%}")
        logger.info(f"Within 0.5 stars: {metrics['within_half_star']:.2%}")
        logger.info(f"Within 1.0 stars: {metrics['within_one_star']:.2%}")
        
        # Log total time
        total_time = time.time() - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        
        logger.info(f"Total pipeline time: {hours}h {minutes}m {seconds}s ({total_time:.2f} seconds)")
        
        return model, mapping_info, metrics
        
    except Exception as e:
        logger.error(f"Error in pipeline: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

if __name__ == "__main__":
    results = full_pipeline()