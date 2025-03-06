### This file runs MF + KNN regressor to adjust low/high ratings and yields better results than base MF
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import gc,time,psutil,functools
import re
from typing import Iterator, Dict, Tuple, List, Optional
import logging
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import torch, torchvision, torchgen, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader, DataChunk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsRegressor

#Basic logging
logging.basicConfig(level = logging.INFO, format='%(asctime)s- %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#to utilize gpu support
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
logging.info(f"Using device {device}")


#New class to help prepare the dataset for the matrix factorization class + apply temporal feature for tuning the model:
class data_for_mf(Dataset):
    def __init__(self, movie_ratings, ratings_scale = 5.0):

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

        #extract movie release years from titles
        release_years = movie_ratings['title'].str.extract(r'\((\d{4})\)').astype(float).values.flatten()

        #calculate age and handle potential negative ages (in case of data errors)
        movie_ages = ratings_years - release_years
        movie_ages = np.maximum(0, movie_ages)  #ensure no negative ages

        #normalize ages into reasonable buckets for embedding layers (0-5 years, 5-10 years, etc)
        age_buckets = np.floor(movie_ages /5).astype(int)   #group into 5 year buckets
        self.age_buckets = torch.LongTensor(age_buckets)

        #keep track of number of unique age buckets for embedding layer
        self.n_age_buckets = age_buckets.max()+1

    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, index):
        return {
            'userID': self.userID[index],
            'movieID': self.movieID[index],
            'rating': self.ratings[index],
            'age_bucket' : self.age_buckets[index]
        }


#Our own custom weight loss to help solve the prediction inaccuracy for ratings below 2 and above 4.5. This aims to predicte ratings that are 1/5 much closer to the actual ratings
class Weightratingloss(nn.Module):
    def __init__(self, alpha = 3.0, beta = 2.0, low_rating_boost = 4.0, high_rating_boost = 2.0):
        super().__init__()
        self.alpha = alpha  #weight for extreme ratings
        self.beta = beta    #Power factor to emphasize the extremes
        self.low_rating_boost = low_rating_boost    #Extra weight for low ratings
        self.high_rating_boost = high_rating_boost  #Extra weight for high ratings

    def forward(self, predictions, targets):
        #Basic error calculation - absolute difference
        base_loss = torch.abs(predictions - targets)

        #calculate how extreme the target rating is (distance from the middle)
        rating_extremity = torch.pow(torch.abs(targets - 0.5), self.beta)

        # Add extra weight for low/high ratings (below 0.3 normalized, or below 1.5 stars)
        low_rating_mask = targets < 0.3 #below 1.5 stars
        high_rating_mask = targets > 0.9  #Above 4.5 stars

        rating_factor = torch.ones_like(targets)
        rating_factor[low_rating_mask] = self.low_rating_boost
        rating_factor[high_rating_mask] = self.high_rating_boost

        # Final weighted loss combines all factors:
        # 1. Base error (absolute difference)
        # 2. General extremity weighting (distance from middle)
        # 3. Special boost for very low/high ratings
        weighted_loss = base_loss * (1+self.alpha *rating_extremity)*rating_factor

        return weighted_loss.mean()

#Matrix factorization for decomposition with Adaptive Moment Estimiation (ADAM) 
class matrix_fact(nn.Module):
    def __init__(self, n_users, n_movies, n_age_buckets , n_factors=100):
        super().__init__()
        """
        Create the parent class/function to perform the matrix factorization:
        parameters:
            - n_users: the users who we will embedd for the model
            - n_movies: the movies we will embedd for the model
        """
        self.logger = logging.getLogger(__name__)
        
        #create the embeddings (unique profiles)
        self.user_factors  = nn.Embedding(n_users, n_factors)
        self.movie_factors = nn.Embedding(n_movies, n_factors)

        self.age_factors = nn.Embedding(n_age_buckets, n_factors) #temporal feature for age buckets

        self.dropout = nn.Dropout(0.2)  #Add dropout for more regularization

        # Add normalization and dropout
        self.user_norm = nn.LayerNorm(n_factors)
        self.movie_norm = nn.LayerNorm(n_factors)
        self.age_norm = nn.LayerNorm(n_factors)
    
        #initilize the embeddings with Xavier/Glorot normal initialiations
        nn.init.xavier_normal_(self.user_factors.weight)
        nn.init.xavier_normal_(self.movie_factors.weight)
        nn.init.xavier_normal_(self.age_factors.weight)

        #create the bias terms, to help the model (adjustment factors that help capture basic tendencies)
        self.user_bias = nn.Embedding(n_users, 1)
        self.movie_bias = nn.Embedding(n_movies, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

        self.to(device)
    
    #the predicition function
    def forward(self, user_ids, movie_ids, age_bucket_ids):
        
        #get the latent factor for each user and movie
        users = self.user_factors(user_ids)
        movies = self.movie_factors(movie_ids)
        ages = self.age_factors(age_bucket_ids)

        #Applying ReLu activaction to help us capture positive interactions more effectively
        users = torch.relu(users)
        movies = torch.relu(movies)
        ages = torch.relu(ages)

        #Applying layer normalization: constant adjustments, to keep the positive values well behaved
        users = self.dropout(self.user_norm(torch.relu(users)))
        movies = self.movie_norm(movies)
        ages = self.age_norm(ages)

        user_bias = self.user_bias(user_ids).squeeze()
        movie_bias = self.movie_bias(movie_ids).squeeze()

        #calculate for dot product now:
        
        movie_with_ages = movies * ages
        combined_features = users * movie_with_ages

        dot_product = combined_features.sum(dim=1)

        predicitions = dot_product / torch.sqrt(torch.tensor(users.shape[1]).float())   #combine all components and scale by embedding dimension for numerical stability
        predicitions = predicitions + user_bias + movie_bias + self.global_bias
        
        predicitions = torch.clamp(predicitions, -0.1, 1.1)  #Clamp predictions to our slightly extended range

        return predicitions
    
    #train the model now
    def train_model(movie_ratings, n_factors = 30, n_epochs = 50, batch_size= 50):
        
        #create mapping dictonaries for reindexing: This is needed for the embeddings
        user_mapping = {old_id: new_id for new_id, old_id in 
                        enumerate(movie_ratings['userID'].unique())}
        movie_mapping = {old_id: new_id for new_id, old_id in
                         enumerate(movie_ratings['movieID'].unique())}     
        
        #create a copy and reindex:
        movie_ratings = movie_ratings.copy()
        movie_ratings['userID'] = movie_ratings['userID'].map(user_mapping)
        movie_ratings['movieID'] = movie_ratings['movieID'].map(movie_mapping)

        #now use the reindexed counts
        n_users = len(user_mapping)
        n_movies = len(movie_mapping)

        #Create dataset to get number of age buckets
        dataset = data_for_mf(movie_ratings, ratings_scale=5.0)
        n_age_buckets = dataset.n_age_buckets

        #initilize the model with age of buckets parameter
        model = matrix_fact(n_users, n_movies, n_age_buckets, n_factors)
        optimizer = optim.Adam(model.parameters(), lr = 0.005, weight_decay = 0.0001)

        #Use the huberloss to handle outliers 
        criterion = Weightratingloss(alpha=5.0, beta=3.0)
        ratings_scale = 5.0   #used to bring the ratings to 0-1, rather than 1-5, easier for the model to understand

        #create dataset/dataloader
        dataset = data_for_mf(movie_ratings, ratings_scale)
        dataloader = DataLoader(dataset, batch_size = 4096, shuffle = True)

        
        #keep track of best model/loss
        best_model_state = None
        best_loss = float('Inf')

        for epoch in range(n_epochs):
            total_loss = 0
            total_count = 0

            for batch in dataloader:
                users = batch['userID'].to(device)
                movies = batch['movieID'].to(device)
                ratings = batch['rating'].to(device)
                age_buckets = batch['age_bucket'].to(device)


                #clear our the previous gradients
                optimizer.zero_grad()

                #forward pass
                predicitions = model(users, movies, age_buckets)

                #calculate loss
                loss = criterion(predicitions, ratings)   # if our model predicts 0.8 even when rating is 1.0, mse = (1-0.8)² = 0.04

                #backward propagtion
                loss.backward()

                #update the optimizer now
                optimizer.step()

                total_loss += loss.item()
                total_count += 1
            
            if (epoch + 1)%5 == 0:
                #convert loss back to original rating scale for interpeliety
                scaled_loss = (total_loss / total_count) * (ratings_scale **2)     

                logging.info(f" Epoch {epoch+1}/{n_epochs} Loss: {scaled_loss:.4f}")

            #save the best model
            if total_loss < best_loss:
                best_loss = loss.item()
                best_model_state = model.state_dict().copy()

        model.load_state_dict(best_model_state) #now load that
        return model, ratings_scale, (user_mapping, movie_mapping)


def calculate_prediction_residuals(model, data, ratings_scale, user_mapping = None, movie_mapping = None):
    """ Calculate prediction residuals for the entire dataset

        Args: trained MF model, data and ratings scale, 
        return: numpy array of residuals
    """

    if user_mapping is None or movie_mapping is None:
        user_mapping = {old_id: new_id for new_id, old_id in 
                       enumerate(data['userID'].unique())}
        movie_mapping = {old_id: new_id for new_id, old_id in
                        enumerate(data['movieID'].unique())}
        
    # Ensure data is reindexed
    user_mapping = {old_id: new_id for new_id, old_id in 
                    enumerate(data['userID'].unique())}
    movie_mapping = {old_id: new_id for new_id, old_id in
                     enumerate(data['movieID'].unique())}
    
    # Create a copy and reindex
    data_reindexed = data.copy()
    data_reindexed['userID'] = data_reindexed['userID'].map(lambda x: user_mapping.get(x, -1))
    data_reindexed['movieID'] = data_reindexed['movieID'].map(lambda x: movie_mapping.get(x, -1))
    

    #drop invalid mappings
    invalid_mask = (data_reindexed['userID'] == -1) | (data_reindexed['movieID'] == -1)
    if invalid_mask.sum() > 0:
        logger.warning(f"Dropping {invalid_mask.sum()} rows with invalid mappings in residuals")
        data_reindexed = data_reindexed[~invalid_mask]

    #prepare dataset
    dataset = data_for_mf(data_reindexed, ratings_scale)
    dataloader = DataLoader(dataset, batch_size= 4096, shuffle=True)

    all_residuals = []

    with torch.no_grad():
        for batch in dataloader:
            users = batch['userID'].to(device)
            movies = batch['movieID'].to(device)
            ratings = batch['rating'].to(device)
            ages = batch['age_bucket'].to(device)

            #get model predictions
            predictions = model(users, movies, ages)
            residuals = (ratings - predictions).cpu().numpy()   #calculate residuals and send to the cpu
            all_residuals.append(residuals)

    return np.concatenate(all_residuals)

def extract_temporal_features(data): 
    """ extract and normalize temporal features"""

    #extract timestamp feature
    timestamps = pd.to_datetime(data['timestamp'])

    features = {
        'year' : timestamps.dt.year,
        'month' : timestamps.dt.month,
        'day_of_week' : timestamps.dt.dayofweek 
    }

    #normalize features
    scaler = MinMaxScaler()

    normalized_features = np.column_stack([
        scaler.fit_transform(feature.values.reshape(-1,1)).flatten()
        for feature in features.values()
    ])

    return normalized_features


def knn_feature_vectors(model, data, ratings_scale, user_mapping, movie_mapping):
    """ Create feature vectors for knn corrections 
        
        args: model, data and ratings_scale
        return: dict of feature components
     """
    

    # Create a copy and reindex
    data_reindexed = data.copy()
    data_reindexed['userID'] = data_reindexed['userID'].map(lambda x: user_mapping.get(x, -1))
    data_reindexed['movieID'] = data_reindexed['movieID'].map(lambda x: movie_mapping.get(x, -1))

    # Drop any invalid mappings
    invalid_mask = (data_reindexed['userID'] == -1) | (data_reindexed['movieID'] == -1)
    if invalid_mask.sum() > 0:
        logger.warning(f"Dropping {invalid_mask.sum()} rows with invalid mappings")
        data_reindexed = data_reindexed[~invalid_mask]

    #get max indices from model embedding dimensions to prevent index errors
    max_user_id = model.user_factors.weight.shape[0] - 1
    max_movie_id = model.movie_factors.weight.shape[0] - 1

    #ensure indices are valid integers and within bounds
    user_ids = np.clip(data_reindexed['userID'].values.astype(int), 0, max_user_id)
    movie_ids = np.clip(data_reindexed['movieID'].values.astype(int), 0, max_movie_id)

    #extract embeddings safely
    user_embeddings = model.user_factors.weight.detach().cpu().numpy()[user_ids]
    movie_embeddings = model.movie_factors.weight.detach().cpu().numpy()[movie_ids]

    try:

        #calculate residuals
        residuals = calculate_prediction_residuals(model, data_reindexed, ratings_scale)
    
    except Exception as e:
        logger.error(f" Error calculating residuals {str(e)}")
        residuals = np.zeros(len(data_reindexed))   #fall back to zero

    #extract temporal features 
    temporal_features = extract_temporal_features(data_reindexed)

    return {
        'user_embeddings': user_embeddings,
        'movie_embeddings': movie_embeddings,
        'residuals': residuals,
        'temporal_features' : temporal_features
    }


def detailed_knn_features(model, train_data, ratings_scale, user_mapping = None, movie_mapping = None):
    """ comprehensive feature prepartion for KNN
    
        args: model, trained_data, ratings_scale
        return: prepared feature matrix for KNN
    """
    if user_mapping is None or movie_mapping is None:
        # Create mapping dictionaries
        user_mapping = {old_id: new_id for new_id, old_id in 
                       enumerate(train_data['userID'].unique())}
        movie_mapping = {old_id: new_id for new_id, old_id in
                        enumerate(train_data['movieID'].unique())}

    # Reindex the data
    train_data_reindexed = train_data.copy()
    train_data_reindexed['userID'] = train_data_reindexed['userID'].map(user_mapping)
    train_data_reindexed['movieID'] = train_data_reindexed['movieID'].map(movie_mapping)

    # Ensure all mappings are valid
    train_data_reindexed = train_data_reindexed.dropna(subset=['userID', 'movieID'])

    #extract feature components from above functions
    feature_components = knn_feature_vectors(model, train_data_reindexed, ratings_scale, user_mapping, movie_mapping)

    #combine features
    combined_features = np.hstack([
        feature_components['user_embeddings'],
        feature_components['movie_embeddings'],
        feature_components['residuals'].reshape(-1,1),
        feature_components['temporal_features']
    ])

    #standarize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(combined_features)

    return scaled_features, feature_components, scaler


class KNNresidualcorrector:
    def __init__(self, n_neighbors = 15, weights = 'distance'):

        """ Initialize Knn residual corrector
            Args: number of neighbours to use,  weights (str): weight function used in prediction
        """
        self.knn = KNeighborsRegressor(
            n_neighbors = n_neighbors,
            weights = weights
        )

        self.scaler = StandardScaler()

        #hyperparameters
        self.n_neighbors = n_neighbors
        self.weight = weights


    def fit(self, features, residuals):
        """ Fit KNN model to residuals we calculated 
            Args: features (prepared feature matrix), residuals (prediction residuals)

            Returns: Fitted model instance
        """
        
        #intialize and scale features
        #scaler = StandardScaler()
        scaled_features = self.scaler.fit_transform(features)

        #fit knn model to residuals
        self.knn.fit(scaled_features, residuals)

        return self
    
    def predict(self, features):
        """ Predict residual corrections
            Args: features (feature matrix for prediction)
            
            returns: predicted residual correction
            """
        
        #if self.scaler is None:
        #    raise ValueError("KNN corrector has not been fitted. Call fit() first")
        
        #scale features
        scaled_features = self.scaler.transform(features)

        #predict residual corrections
        return self.knn.predict(scaled_features)
    
    def apply_correction(self, model, train_data, test_data, ratings_scale):
        """ Apply knn-based corrector to matrix factoriazation predictions

            Args: model(trained model), train_data, test_data, ratings_scale
            
            returns: knn corrections for test data"""

        train_features, _, _ = detailed_knn_features(model, train_data, ratings_scale)
        train_residuals = calculate_prediction_residuals(model, train_data, ratings_scale)

        #fit corrector on training set
        self.fit(train_features, train_residuals)

        #prepare test features
        test_features, _, _ = detailed_knn_features(model, test_data, ratings_scale)

        return self.predict(test_features)  #predict the corrections


#lets evaulate the model and test on unseen data
def evaluate_model(model, test_data, user_mapping, movie_mapping, ratings_scale, knn_corrector, max_chunk_size = 100000, batch_size = 512):
    
    """
    Evaluate the model performance on unseen test data

    Arguments:
        model: train matrix factoriazation model
        test_data: Dataframe containing test data
        user_mapping: dict mapping userID to indices
        movie_mapping: dict mapping movieID to indicies
        ratings_scale: Scale factor for adjusting ratings

    Returns:
        dict of evaluation metrics
    """
    
    # First, remap the test data IDs to match training indexes
    # This is the key fix - we need to remap before chunking
    test_data_remapped = test_data.copy()
    test_data_remapped['userID'] = test_data_remapped['userID'].map(user_mapping)
    test_data_remapped['movieID'] = test_data_remapped['movieID'].map(movie_mapping)
    
    # Drop any rows with invalid mappings right at the start
    invalid_mask = test_data_remapped['userID'].isna() | test_data_remapped['movieID'].isna()
    if invalid_mask.sum() > 0:
        logger.warning(f"Dropping {invalid_mask.sum()} rows with invalid mappings from test data")
        test_data_remapped = test_data_remapped[~invalid_mask]
    
    # Now continue with the chunking as before, but on remapped data
    test_chunks = []
    for i in range(0, len(test_data_remapped), max_chunk_size):
        end_idx = min(i + max_chunk_size, len(test_data_remapped))
        test_chunks.append(test_data_remapped.iloc[i:end_idx])
    
    all_preds = []
    all_actuals = []

    logger.info(f"Evaluating in {len(test_chunks)} chunks of max size {max_chunk_size}")
    
    for chunk_idx, chunk in enumerate(test_chunks):
        logger.info(f"Processing evaluation chunk {chunk_idx+1}/{len(test_chunks)}")

        #prepare dataset for this chunk only
        dataset = data_for_mf(chunk, ratings_scale)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        chunk_preds = []
        chunk_actuals = []        
    
        #process batches within this chunk
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                try:
                    users = batch['userID'].to(device)
                    movies = batch['movieID'].to(device)
                    ratings = batch['rating'].to(device)
                    ages = batch['age_bucket'].to(device)

                    base_pred = model(users, movies, ages)  #get base predictions

                    base_pred_np = base_pred.cpu().numpy()

                    #get the corresponding slice from the original chunk for KNN features
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, len(chunk))
                    batch_df = chunk.iloc[start_idx: end_idx].copy()

                    #ensure batch_df has same lengths as batch tensors
                    if len(batch_df) != len(users):
                        logger.warning(f"Batch size mismatch: {len(batch_df)} vs {len(users)}. Skipping KNN")
                        corrected_pred_np = base_pred_np
                    else:
                        # Since data is already remapped, we can pass empty mappings here
                        # to prevent double-mapping
                        empty_user_mapping = {id: id for id in batch_df['userID'].unique()}
                        empty_movie_mapping = {id: id for id in batch_df['movieID'].unique()}
                        
                        #prepare knn features - with identity mappings since data is already mapped
                        batch_features,_,_ = detailed_knn_features(model, batch_df, ratings_scale, 
                                                                  empty_user_mapping, empty_movie_mapping)

                        #apply KNN corrections
                        knn_corrections = knn_corrector.predict(batch_features)

                        # Apply differential correction based on prediction range
                        corrected_pred_np = base_pred_np.copy()

                        #apply full correction for extreme predictions (below 2.0 and above 4.0)
                        low_mask = base_pred_np * ratings_scale < 2.0
                        high_mask = base_pred_np * ratings_scale > 4.0
                        mid_mask = ~(low_mask | high_mask)

                        #apply different correction strengths based on prediction range
                        corrected_pred_np[low_mask] += knn_corrections[low_mask]    #Full correction for low ratings
                        corrected_pred_np[high_mask] += knn_corrections[high_mask]  #full correction for medium ratings
                        corrected_pred_np[mid_mask] += 0.3 *knn_corrections[mid_mask] #reduced correction for middle

                        #store both base and corrected predictions
                        chunk_preds.append(corrected_pred_np *ratings_scale)
                        chunk_actuals.append((ratings * ratings_scale).cpu().numpy())

                except Exception as e:
                    logger.error(f"Error in batch processing, {batch_idx}: {str(e)}")

        #combine chunk results and add to the overall result
        chunk_preds_np = np.concatenate(chunk_preds)
        chunk_actuals_np = np.concatenate(chunk_actuals)

        all_preds.append(chunk_preds_np)
        all_actuals.append(chunk_actuals_np)

        #clear chunk data to free memory
        del chunk_preds,chunk_actuals, chunk_preds_np, chunk_actuals_np
        gc.collect()    #force garbage collection

    #combine all chunks
    predictions_np = np.concatenate(all_preds)
    actuals_np = np.concatenate(all_actuals)
    
    # Calculate metrics on CPU
    mae = np.mean(np.abs(predictions_np - actuals_np))
    rmse = np.sqrt(np.mean((predictions_np - actuals_np) ** 2))
  
    #Round predictions to nearst 0.5 for display purposes
    rounds_pred = np.round(predictions_np *2)/2

    #calculate accuracy metrics
    exact_match = np.mean(rounds_pred == actuals_np)
    within_half_star = np.mean(np.abs(rounds_pred - actuals_np) <= 0.5)
    within_one_star = np.mean(np.abs(rounds_pred - actuals_np) <= 1.0)

    # Display sample predictions
    logger.info("Sample predictions from test set: ")
    sample_indices = np.random.choice(len(predictions_np), min(20, len(predictions_np)), replace=False)

    for idx in sample_indices:
        logger.info(f"Predicted: {rounds_pred[idx]:.1f}, Actual: {actuals_np[idx]:.1f}, "
            f"Error: {abs(rounds_pred[idx] - actuals_np[idx]):.1f}")
    
    #Summarize performance by rating category:
    rating_categories = [1.0, 2.0, 3.0, 4.0, 5.0]
    logger.info("Performace by rating category: ")
    for rating in rating_categories:
        mask = np.isclose(actuals_np, rating)
        if np.sum(mask) >0:
            category_mae = np.mean(np.abs(predictions_np[mask] - actuals_np[mask]))
            logger.info(f"Rating {rating:.1f}: MAE = {category_mae:.4f}, Count = {np.sum(mask)}")
        
    #Return all metrics
    metrics = {
        'mae':mae,
        'rmse': rmse,
        'exact_match': exact_match,
        'within_half_star':within_half_star,
        'within_one_star':within_one_star
    }

    logger.info(f" Overall metrics - MAE : {mae:.4f}, RMSE: {rmse:.4f}")
    logger.info(f"Accuracy - Exact: {exact_match:.2%}, Within 0.5: {within_half_star:.2%}, "
            f"Within 1.0: {within_one_star:.2%}")

    return metrics


def main():
    """
    Main function to load and process all data files. """

    #Lets apply the train/test evaluation
    start_time = time.time()
    try:
        # 1. Load processed data
        logger.info("Initializing data loader...")
        base_path = Path("data/processed")

        logger.info(f"Loading data from parquet files")
        
        # Read partitioned parquet data
        movie_ratings = pd.read_parquet(str(base_path / "movie_ratings"),engine='pyarrow')
            
        logger.info(f"Original Dataset size: {len(movie_ratings):,}")

        # 2. Start with a small subset to verify pipeline works
        sample_size = 3500000  
        logger.info(f"Creating proof-of-concept with {sample_size:,} samples")
        
        sample_data = movie_ratings.sample(n=sample_size, random_state=42)
        train_data, test_data = train_test_split(sample_data, test_size=0.2, random_state=42)

        # Create mapping dictionaries for users and movies in the training data
        user_mapping = {old_id: new_id for new_id, old_id in 
                enumerate(train_data['userID'].unique())}
        movie_mapping = {old_id: new_id for new_id, old_id in
                        enumerate(train_data['movieID'].unique())}
        
        # Filter test data to only include users and movies in the training set
        train_users = set(train_data['userID'].unique())
        train_movies = set(train_data['movieID'].unique())
        
        filtered_test_data = test_data[  
                test_data['userID'].isin(train_users) &  
                test_data['movieID'].isin(train_movies)  
            ] 
        
        logger.info(f"Training set size: {len(train_data):,}")
        logger.info(f"Testing set size: {len(test_data):,}")
        logger.info(f"Filtered test size {len(filtered_test_data):,}")

        # 3. Train model on training data only, using our pre-created mapping dictionaries
        logger.info("Starting model training...")
        
        # First, remap the training data
        train_data_remapped = train_data.copy()
        train_data_remapped['userID'] = train_data_remapped['userID'].map(user_mapping)
        train_data_remapped['movieID'] = train_data_remapped['movieID'].map(movie_mapping)
        
        # Train model with remapped data
        model, ratings_scale, _ = matrix_fact.train_model(
                train_data_remapped,
                n_factors=128,
                n_epochs=20, 
                batch_size=4096
            )

        #4. save model/mappings
        save_path = Path("model")
        save_path.mkdir(exist_ok=True)
        torch.save(model.state_dict(), save_path / "movie_recommender_model.pth")

        #save these mappings for future use:
        mapping_info = {
                'user_mapping': user_mapping,
                'movie_mapping': movie_mapping,
                'ratings_scale': ratings_scale
            }
        torch.save(mapping_info, save_path / "mapping_info.pth")

        #5. Evaluate the model on the test data...
        logger.info("Evaluating the model on test data...")
        
        # Create empty mappings for the already-mapped data
        id_mapping_train = {id: id for id in range(max(train_data_remapped['userID'].max(), 
                                                      train_data_remapped['movieID'].max())+1)}
        
        features, _, _ = detailed_knn_features(model, train_data_remapped, ratings_scale, 
                                              id_mapping_train, id_mapping_train)
        residuals = calculate_prediction_residuals(model, train_data_remapped, ratings_scale, 
                                                  id_mapping_train, id_mapping_train)

        knn_corrector = KNNresidualcorrector()
        knn_corrector.fit(features, residuals)
        
        metrics = evaluate_model(model, filtered_test_data, user_mapping, movie_mapping, 
                                 ratings_scale, knn_corrector, max_chunk_size=5000, 
                                 batch_size=4096)

        #6. Total processing Time        
        total_time = time.time() - start_time
        logger.info(f" Total Processing time: {total_time:.2f} seconds")

        return model, mapping_info, metrics
    
    except Exception as e:
        logger.error(f"Error in training model on pipeline: {str(e)}")
        logger.exception(e)  # Add full traceback for better debugging
        return None

if __name__ == '__main__':
    final_result = main()
   