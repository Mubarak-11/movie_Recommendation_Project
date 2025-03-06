### model_definition.py - Contains the Matrix Factorization model definition and supporting classes

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import gc
import time
from data_processing_part_1 import DataProcess

# Basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s- %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set device for PyTorch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device {DEVICE}")

class DataForMF(Dataset):
    """Dataset class for Matrix Factorization model"""
    def __init__(self, movie_ratings, ratings_scale=5.0):
        # Make sure we're working with valid numeric IDs before creating tensors
        # Ensure no NaN values
        movie_ratings = movie_ratings.dropna(subset=['userID', 'movieID', 'rating'])
        
        # Convert to integers to avoid potential float->long conversion issues
        try:
            self.userID = torch.LongTensor(movie_ratings['userID'].astype(int).values)
            self.movieID = torch.LongTensor(movie_ratings['movieID'].astype(int).values)
            self.ratings = torch.FloatTensor(movie_ratings['rating'].values) / ratings_scale
        except Exception as e:
            logger.error(f"Error converting IDs to tensors: {e}")
            # Provide more details for debugging
            logger.error(f"userID types: {movie_ratings['userID'].dtype}")
            logger.error(f"movieID types: {movie_ratings['movieID'].dtype}")
            logger.error(f"Sample userIDs: {movie_ratings['userID'].head()}")
            logger.error(f"Sample movieIDs: {movie_ratings['movieID'].head()}")
            raise
    
        # Calculate the movie age at the time of ratings
        rating_timestamp = pd.to_datetime(movie_ratings['timestamp'])
        ratings_years = rating_timestamp.dt.year.values

        # Extract movie release years from titles
        release_years = movie_ratings['title'].str.extract(r'\((\d{4})\)').astype(float).values.flatten()

        # Calculate age and handle potential negative ages (in case of data errors)
        movie_ages = ratings_years - release_years
        movie_ages = np.maximum(0, movie_ages)  # Ensure no negative ages

        # Normalize ages into reasonable buckets for embedding layers (0-5 years, 5-10 years, etc)
        age_buckets = np.floor(movie_ages / 5).astype(int)  # Group into 5 year buckets
        self.age_buckets = torch.LongTensor(age_buckets)

        # Keep track of number of unique age buckets for embedding layer
        self.n_age_buckets = int(age_buckets.max() + 1)

    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, index):
        return {
            'userID': self.userID[index],
            'movieID': self.movieID[index],
            'rating': self.ratings[index],
            'age_bucket': self.age_buckets[index]
        }

class WeightRatingLoss(nn.Module):
    """
    Custom weight loss to help solve the prediction inaccuracy for ratings below 2 and above 4.5.
    This aims to predict extreme ratings (1 and 5) much closer to the actual ratings.
    """
    def __init__(self, alpha=3.0, beta=2.0, low_rating_boost=5.0, high_rating_boost=2.0):
        super().__init__()
        self.alpha = alpha  # Weight for extreme ratings
        self.beta = beta    # Power factor to emphasize the extremes
        self.low_rating_boost = low_rating_boost    # Extra weight for low ratings
        self.high_rating_boost = high_rating_boost  # Extra weight for high ratings

    def forward(self, predictions, targets):
        # Basic error calculation - absolute difference
        base_loss = torch.abs(predictions - targets)

        # Calculate how extreme the target rating is (distance from the middle)
        rating_extremity = torch.pow(torch.abs(targets - 0.5), self.beta)

        # Add extra weight for low/high ratings (below 0.3 normalized, or below 1.5 stars)
        low_rating_mask = targets < 0.3  # Below 1.5 stars
        high_rating_mask = targets > 0.9  # Above 4.5 stars

        rating_factor = torch.ones_like(targets)
        rating_factor[low_rating_mask] = self.low_rating_boost
        rating_factor[high_rating_mask] = self.high_rating_boost

        # Final weighted loss combines all factors:
        # 1. Base error (absolute difference)
        # 2. General extremity weighting (distance from middle)
        # 3. Special boost for very low/high ratings
        weighted_loss = base_loss * (1 + self.alpha * rating_extremity) * rating_factor

        return weighted_loss.mean()

class MatrixFact(nn.Module):
    """
    Matrix factorization for recommendation system with temporal feature (movie age).
    Uses adaptive moment estimation (ADAM) for optimization.
    """
    def __init__(self, n_users, n_movies, n_age_buckets, n_factors=100):
        super().__init__()
        """
        Create the parent class/function to perform the matrix factorization:
        parameters:
            - n_users: the users who we will embed for the model
            - n_movies: the movies we will embed for the model
            - n_age_buckets: buckets of movie ages for temporal embedding
            - n_factors: dimensionality of the latent factors
        """
        self.logger = logging.getLogger(__name__)
        
        # Create the embeddings (unique profiles)
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.movie_factors = nn.Embedding(n_movies, n_factors)
        self.age_factors = nn.Embedding(n_age_buckets, n_factors)  # Temporal feature

        # Add regularization
        self.dropout = nn.Dropout(0.2)

        # Add normalization
        self.user_norm = nn.LayerNorm(n_factors)
        self.movie_norm = nn.LayerNorm(n_factors)
        self.age_norm = nn.LayerNorm(n_factors)
    
        # Initialize the embeddings with Xavier/Glorot normal initializations
        nn.init.xavier_normal_(self.user_factors.weight)
        nn.init.xavier_normal_(self.movie_factors.weight)
        nn.init.xavier_normal_(self.age_factors.weight)

        # Create the bias terms (adjustment factors that help capture basic tendencies)
        self.user_bias = nn.Embedding(n_users, 1)
        self.movie_bias = nn.Embedding(n_movies, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

        # Move model to device (GPU if available)
        self.to(DEVICE)
    
    def forward(self, user_ids, movie_ids, age_bucket_ids):
        """
        Forward pass to predict ratings
        """
        # Get the latent factor for each user and movie
        users = self.user_factors(user_ids)
        movies = self.movie_factors(movie_ids)
        ages = self.age_factors(age_bucket_ids)

        # Apply ReLU activation to help capture positive interactions more effectively
        users = torch.relu(users)
        movies = torch.relu(movies)
        ages = torch.relu(ages)

        # Apply layer normalization and dropout for regularization
        users = self.dropout(self.user_norm(torch.relu(users)))
        movies = self.movie_norm(movies)
        ages = self.age_norm(ages)

        # Get bias terms
        user_bias = self.user_bias(user_ids).squeeze()
        movie_bias = self.movie_bias(movie_ids).squeeze()

        # Calculate interactions with temporal features
        movie_with_ages = movies * ages
        combined_features = users * movie_with_ages

        # Calculate dot product
        dot_product = combined_features.sum(dim=1)

        # Final prediction combining all components
        predictions = dot_product / torch.sqrt(torch.tensor(users.shape[1]).float())
        predictions = predictions + user_bias + movie_bias + self.global_bias
        
        # Clamp predictions to slightly extended range
        predictions = torch.clamp(predictions, -0.1, 1.1)

        return predictions
    
    @staticmethod
    def train_model(movie_ratings, n_factors=30, n_epochs=50, batch_size=50):
        """
        Train the matrix factorization model
        
        Args:
            movie_ratings: DataFrame with movie ratings
            n_factors: Number of latent factors
            n_epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            model: Trained model
            ratings_scale: Scale factor for ratings
            mapping_tuple: Tuple of (user_mapping, movie_mapping)
        """
        from torch.utils.data import DataLoader
        
        # Create mapping dictionaries for reindexing (needed for embeddings)
        user_mapping = {old_id: new_id for new_id, old_id in 
                        enumerate(movie_ratings['userID'].unique())}
        movie_mapping = {old_id: new_id for new_id, old_id in
                         enumerate(movie_ratings['movieID'].unique())}     
        
        # Create a copy and reindex
        movie_ratings = movie_ratings.copy()
        movie_ratings['userID'] = movie_ratings['userID'].map(user_mapping)
        movie_ratings['movieID'] = movie_ratings['movieID'].map(movie_mapping)

        # Get counts after reindexing
        n_users = len(user_mapping)
        n_movies = len(movie_mapping)

        # Create dataset to get number of age buckets
        dataset = DataForMF(movie_ratings, ratings_scale=5.0)
        n_age_buckets = dataset.n_age_buckets

        # Initialize the model with age buckets parameter
        model = MatrixFact(n_users, n_movies, n_age_buckets, n_factors)
        optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0001)

        # Use the custom loss to handle extreme ratings
        criterion = WeightRatingLoss(alpha=5.0, beta=3.0)
        ratings_scale = 5.0   # Used to normalize ratings to 0-1 range

        # Create dataset/dataloader
        dataset = DataForMF(movie_ratings, ratings_scale)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Keep track of best model/loss
        best_model_state = None
        best_loss = float('inf')

        for epoch in range(n_epochs):
            total_loss = 0
            total_count = 0

            for batch in dataloader:
                users = batch['userID'].to(DEVICE)
                movies = batch['movieID'].to(DEVICE)
                ratings = batch['rating'].to(DEVICE)
                age_buckets = batch['age_bucket'].to(DEVICE)

                # Clear previous gradients
                optimizer.zero_grad()

                # Forward pass
                predictions = model(users, movies, age_buckets)

                # Calculate loss
                loss = criterion(predictions, ratings)

                # Backward propagation
                loss.backward()

                # Update parameters
                optimizer.step()

                total_loss += loss.item()
                total_count += 1
            
            if (epoch + 1) % 5 == 0:
                # Convert loss back to original rating scale for interpretability
                scaled_loss = (total_loss / total_count) * (ratings_scale ** 2)     
                logging.info(f"Epoch {epoch+1}/{n_epochs} Loss: {scaled_loss:.4f}")

            # Save the best model
            if total_loss < best_loss:
                best_loss = total_loss
                best_model_state = model.state_dict().copy()

        # Load the best model state
        model.load_state_dict(best_model_state)
        return model, ratings_scale, (user_mapping, movie_mapping)

def main():
    """
    Test the model definition with a small dataset
    """
    
    start_time = time.time()
    try:
        # Load a small sample of data for testing
        processor = DataProcess(chunk_size=10000)
        movie_ratings = processor.process_file('Movie_ratings.csv', sep=',')
        logger.info(f"Original dataset size: {len(movie_ratings)}")
        
        # Sample a smaller dataset for testing
        test_sample = movie_ratings.sample(n=10000, random_state=42)
        logger.info(f"Test sample size: {len(test_sample)}")
        
        # Create timestamp column if needed (for age buckets)
        if 'timestamp' in test_sample.columns and not pd.api.types.is_datetime64_any_dtype(test_sample['timestamp']):
            test_sample['timestamp'] = pd.to_datetime(test_sample['timestamp'], unit='s')
        
        # Train model with small number of factors and epochs for testing
        model, ratings_scale, (user_mapping, movie_mapping) = MatrixFact.train_model(
            test_sample,
            n_factors=20,
            n_epochs=5,
            batch_size=256
        )
        
        logger.info(f"Model training completed")
        logger.info(f"Number of users: {len(user_mapping)}")
        logger.info(f"Number of movies: {len(movie_mapping)}")
        
        # Save model for later use
        save_path = Path("model")
        save_path.mkdir(exist_ok=True)
        torch.save(model.state_dict(), save_path / "test_model.pth")
        
        # Save mappings
        mapping_info = {
            'user_mapping': user_mapping,
            'movie_mapping': movie_mapping,
            'ratings_scale': ratings_scale
        }
        torch.save(mapping_info, save_path / "test_mapping_info.pth")
        
        logger.info(f"Model and mappings saved")
        logger.info(f"Total testing time: {time.time() - start_time:.2f} seconds")
        
        return model, mapping_info
        
    except Exception as e:
        logger.error(f"Error testing model: {e}")
        return None

if __name__ == '__main__':
    # This will test the model with a small dataset
    test_results = main()