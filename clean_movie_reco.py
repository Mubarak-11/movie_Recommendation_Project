### Building a movie_recommendation system, Clean and concise, trying to write efficient and easy code

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

#Basic logging
logging.basicConfig(level = logging.INFO, format='%(asctime)s- %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#to utilize gpu support
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device {DEVICE}")

class DataProcess:
    """
    Handles efficient loading and basic processing of movie data files.
    Uses chunking to manage memory when reading large files.
    """
    def __init__(self, chunk_size: int = 500000):
        # Chunk size can be adjusted based on available system memory
        self.chunk_size = chunk_size
    
    def process_file(self, filename: str, sep: str = ',') -> pd.DataFrame:
        """
        Reads and processes a data file in chunks to manage memory efficiently.
        
        Args:
            filename: Path to the data file
            sep: Separator used in the file (default ',' for CSV, '::' for DAT files)
            
        Returns:
            Complete DataFrame after processing all chunks
        """
        chunks = []
        try:
            # Process the file in chunks
            for chunk in pd.read_csv(filename, sep=sep, engine='python', chunksize=self.chunk_size):
                chunks.append(chunk)
            
            # Combine all chunks into a single DataFrame
            return pd.concat(chunks, ignore_index=True)
            
        except Exception as e:
            logger.error(f"Error processing file {filename}: {e}")
            raise

#create a seperate class for feature engineering (numpy operations: vectorization/broadcasting) and data cleaning/prepartion:
class feature_eng_and_data_clean:
    def __init__(self, data):
        self.data = data
    
    #Drop missing rows and feature_engineering 1: extract year from movie title
    def feature_eng_1(self):

        self.data["tags"] = self.data["tags"][self.data["tags"].notna()]  #drop missing rows from tags dataframe
        self.data['movie_ratings']['year'] = self.data['movie_ratings']['title'].str.extract(r"\((\d{4})\)")

        return self.data
    
    #feature_engineering 2: genre column encoding
    def feature_eng_2(self):
        self.genres_encoded = self.data["movie_ratings"]["genres"].str.get_dummies("|")
        return self.genres_encoded
    
    #feature engineering 3: Rating statstics
    def feature_eng_3(self):
        self.rating_stats = self.data["movie_ratings"].groupby(["title"]).agg({
             "rating": ["size", "mean", "median", "std"] #size is the number of ratings
        })

        
        return self.rating_stats

    #feature engineering 4: Extract Timestamps features, converting timestamp column to datetime format
    def feature_eng_4(self):

        self.data["movie_ratings"]["timestamp"] = pd.DatetimeIndex(
            pd.to_datetime(self.data["movie_ratings"]["timestamp"],
                            unit="s", 
                            cache= True
            )
        )
        
        return self.data

#class to utilize the features and plot them with matplotlib/seaborn
class visualizations:
    def __init__(self,data):
        self.data = data
        self.rating_stats = self.data["movie_ratings"].groupby(["title"]).agg({
             "rating": ["size", "mean", "median", "std"] #size is the number of ratings
        })

    def plot(self):

        #plot 1: top 10 most rated movies
        plt.figure(figsize=(10, 6))
        top_movies = self.rating_stats["rating"]["size"].nlargest(10)
        sns.barplot(x = top_movies.values, y = top_movies.index, color="red")
        plt.title("Top 10 Most Rated Movies")
        plt.xlabel("Number of Ratings")
        plt.tight_layout()

        
        #plot 2: User activity vs Average Rating
        plt.figure(figsize=(10,6))
        plt.scatter(self.rating_stats["rating"]["size"], self.rating_stats["rating"]["mean"], edgecolors= "k", color="purple", alpha=0.5)
        plt.title("User Activity vs Average Rating")
        plt.xlabel("Number of Ratings")
        plt.ylabel("Average Rating")
        plt.grid(True)
        plt.tight_layout()

        #plot 3: Distrubtion of Ratings?
        plt.figure(figsize=(10,6))
        sns.histplot(data = self.data["movie_ratings"],x="rating" ,bins = 10, kde= True, color = "green")
        plt.title("Distributions of Ratings")
        plt.ylabel("Rating")
        plt.xlabel("Count")
        plt.tight_layout()


        plt.figure(figsize=(10,8))
        numeric_columns = self.data["movie_ratings"].select_dtypes(include = [np.number])
        self.corrleation_matrix = numeric_columns.corr()
        sns.heatmap(self.corrleation_matrix, annot = True, cmap = "coolwarm", fmt = ".2f", linewidths = 0.5)
        plt.title("Heatmap of Correlation between User-Movie Features")
        plt.tight_layout()

        #plt.show()

#create a seperate class for implementing different recommendtion alogrithms


#create a few classes for building an ETL pipline
class moviedataextractor():
    """ Handles data source connections and raw data extraction"""
    def __init__(self, chunk_size: int = 500000):
        self.chunk_size = chunk_size
        self.logger = logging.getLogger(__name__)

    def validate_data(self, data):
        """ Validate data strucuture/content before extraction"""
        try:
            expected_columns = {
                'movies': ['movieId', 'title', 'genres'],
                'ratings': ['userId', 'movieId', 'rating', 'timestamp'],  
                'tags': ['userId', 'movieId', 'tag', 'timestamp']
            }

            for key, df in data.items():
                 # Add debugging information
                self.logger.info(f"Checking {key} dataframe")
                self.logger.info(f"Expected columns: {expected_columns[key]}")
                self.logger.info(f"Actual columns: {df.columns.tolist()}")
                
                missing_cols = [col for col in expected_columns[key] if col not in df.columns]
                if missing_cols:
                    self.logger.error(f"Missing columns in {key}: {missing_cols}")
            
                assert all(col in df.columns for col in expected_columns[key]), f"Missing columns in {key}"
                assert not df.empty, f"{key} dataframe is empty"
            
            return True
            
        except AssertionError as e:
            self.logger.error(f"Validation failed: {str(e)}")
            raise


    def extract_from_source(self, file_paths):
        start_time = time.time()
        try:    
            # Define column mappings once, outside the loop
            column_mappings = {
                'movies': ['movieId', 'title', 'genres'],
                'ratings': ['userId', 'movieId', 'rating', 'timestamp'],
                'tags': ['userId', 'movieId', 'tag', 'timestamp']
            }
            
            # Define file configurations once
            file_configs = {
                'dat': {'sep': '::', 'header': None},
                'csv': {'sep': ',', 'index_col': 0}
            }
            
            data = {}
            for key, filepath in file_paths.items():
                file_type = filepath.split('.')[-1]
                config = file_configs[file_type].copy()
                
                # Add column names for DAT files
                if file_type == 'dat':
                    config['names'] = column_mappings[key]
                
                # Read data with optimal configuration
                df = pd.concat(
                    pd.read_csv(
                        filepath,
                        engine='python',
                        chunksize=self.chunk_size,
                        **config
                    ),
                    ignore_index=True,
                    copy=False
                )
                
                data[key] = df
                self.logger.info(f"Processed {key}: {df.shape}")
                
            return data
            
        except Exception as e:
            self.logger.error(f"Extraction failed: {str(e)}")
            raise


class moviedatatransform():
    """Handles data transformation and cleaning"""
    def __init__(self):  # Removed data parameter as it's not used
        self.logger = logging.getLogger(__name__)
        self.quality_metrics = {}

    def check_data_quality(self, data, stage="pre"):
        """Check data quality before/after transformation"""
        metrics = {
            'missing_values': data.isnull().sum().to_dict(),
            'unique_counts': {col: data[col].nunique() for col in data.columns},
            'memory_usage': data.memory_usage(deep=True).sum() / 1024**2  # MB
        }
        self.quality_metrics[f"{stage}_transform"] = metrics  # Fixed underscore position
        return metrics

    def transform(self, data):  # Renamed method to match main() call
        """Applies all transformation steps to the data"""
        start_time = time.time()  # Fixed time.time call
        try:
            # Pre-transform quality check
            self.check_data_quality(data["movie_ratings"], "pre")

            transformed_data = {
                'tags': data['tags'][data['tags'].notna()],
                'movie_ratings': data["movie_ratings"].copy()
            }

            # Year extraction
            transformed_data["movie_ratings"]["year"] = (
                transformed_data["movie_ratings"]["title"].str.extract(r"\((\d{4})\)")
            )

            # Genre encoding
            genres_encoded = (
                transformed_data["movie_ratings"]["genres"].str.get_dummies("|")
            )

            # Rating statistics
            rating_stats = (
                transformed_data["movie_ratings"].groupby(["title"]).agg({
                    "rating": ["size", "mean", "median", "std"]
                })
            )

            # Timestamp conversion
            transformed_data["movie_ratings"]["timestamp"] = pd.DatetimeIndex(
                pd.to_datetime(
                    transformed_data["movie_ratings"]["timestamp"],  # Fixed self.data to transformed_data
                    unit="s",
                    cache=True
                )
            )

            # Post-transform quality check
            self.check_data_quality(transformed_data["movie_ratings"], "post")

            self.logger.info(f"Transformation completed in {time.time() - start_time:.2f} seconds")
            return transformed_data, genres_encoded, rating_stats

        except Exception as e:
            self.logger.error(f"Transformation failed: {str(e)}")
            raise

class moviedataload():
    """Handles data storage and retrieval"""
    def __init__(self, storage_path):
        self.storage_path = Path(storage_path)
        self.logger = logging.getLogger(__name__)
        self.storage_path.mkdir(parents=True, exist_ok=True)
    
    def load_data(self, data, partition_cols=None):
        """Load data into parquet format with optional partitioning"""
        start_time = time.time()  # Fixed time.time call
        try:
            for key, df in data.items():
                filepath = self.storage_path / key
                
                # Create directory for this dataset
                filepath.mkdir(parents=True, exist_ok=True)
                
                if partition_cols and partition_cols[0] in df.columns:
                    # If we want to partition and the column exists in this dataset
                    table = pa.Table.from_pandas(df)
                    pq.write_to_dataset(
                        table,
                        root_path=str(filepath),
                        partition_cols=partition_cols,
                        compression="snappy",
                        use_dictionary=True,
                        
                    )
                else:
                    # Regular write without partitioning
                    table = pa.Table.from_pandas(
                        df,
                        preserve_index=False
                    )
                    pq.write_table(
                        table,
                        filepath / "data.parquet",
                        compression="snappy",
                        use_dictionary=True
                    )
                
                self.logger.info(f"Saved {key} dataset with shape {df.shape}")
                
            self.logger.info(f"Data loaded to {self.storage_path} in {time.time() - start_time:.2f} seconds")
        
        except Exception as e:
            self.logger.error(f"Loading failed: {str(e)}")  # Fixed error logging
            raise

    def read_data(self, dataset_name):
        """Reads data from parquet storage"""
        try:
            filepath = self.storage_path / f"{dataset_name}.parquet"
            return pd.read_parquet(filepath)
        except Exception as e:
            self.logger.error(f"Reading failed: {str(e)}")  # Fixed errors to error
            raise

#New class to help prepare the dataset for the matrix factorization class + apply temporal feature for tuning the model:
class data_for_mf(Dataset):
    def __init__(self, movie_ratings, ratings_scale = 5.0):
        self.userID = torch.LongTensor(movie_ratings['userID'].values)
        self.movieID = torch.LongTensor(movie_ratings['movieID'].values)
        self.ratings = torch.FloatTensor(movie_ratings['rating'].values) / ratings_scale
    
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
    def __init__(self, alpha = 3.0, beta = 2.0, low_rating_boost = 3.0, high_rating_boost = 2.0):
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

        self.to(DEVICE)
    
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
        users = self.user_norm(users)
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
        criterion = Weightratingloss(alpha=3.0, beta=2.0)

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
                users = batch['userID'].to(DEVICE)
                movies = batch['movieID'].to(DEVICE)
                ratings = batch['rating'].to(DEVICE)
                age_buckets = batch['age_bucket'].to(DEVICE)


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



def main():
    """
    Main function to load and process all data files.
    Returns a dictionary containing all processed DataFrames.
    """
    start_time = time.time()
    processor = DataProcess(chunk_size=10000)
    data = {} # Dictionary to store all our processed data
    try:
        
        # Load the DAT files
        #logger.info("Processing DAT files...")
        #data['movies'] = processor.process_file('movies.dat', sep='::')
        #data['ratings'] = processor.process_file('ratings.dat', sep='::')
        #data['tags'] = processor.process_file('tags.dat', sep='::')
        
        # Load the main CSV file
        data['movie_ratings'] = processor.process_file('Movie_ratings.csv', sep=',')

        logger.info(f"Original dataset size: {len(data['movie_ratings'])}")
        
        data['movie_ratings'] = data['movie_ratings'].sample(n=50000, random_state=42)  #using a random sample of 50,000 from the movie_ratings
        logger.info(f"Sampled dataset size: {len(data['movie_ratings'])}")
        #movie_ratings = processor.process_file('Movie_ratings.csv', sep=',')
        #print(data['movie_ratings'].head(),"\n")

          # Add debugging info for user and movie IDs
        n_users = data['movie_ratings']['userID'].nunique()
        n_movies = data['movie_ratings']['movieID'].nunique()
        max_user_id = data['movie_ratings']['userID'].max()
        max_movie_id = data['movie_ratings']['movieID'].max()
        
        logger.info(f"Number of unique users: {n_users}")
        logger.info(f"Number of unique movies: {n_movies}")

        processing_time = time.time() - start_time  #calculate duration 
        logger.info(f"Data loading in:  {processing_time:.2f} seconds\n")
        
    except Exception as e:
        logger.error(f"Error in Data Loading: {e}")
        return None
    """
    try:
        #pass the data to the second class: 
        second_class = feature_eng_and_data_clean(data)

        # feature engineering #1
        data = second_class.feature_eng_1() #store the processed data
        #print("The year column contains: ",data["movie_ratings"]["year"].isnull().sum(), " empty rows\n")
        
        #feature engineering #2
        genres_encoded = second_class.feature_eng_2()
        data["movie_ratings"] = pd.concat([data["movie_ratings"], genres_encoded], axis=1) 
        #print(data["movie_ratings"].head())
        
        #feature engineering #3
        rating_stats = second_class.feature_eng_3()
        #print("printing the stats :", rating_stats.head(),"\n")

        #feature engineering #4
        data = second_class.feature_eng_4()
        #print("Converted timestamps: ",data["movie_ratings"]["timestamp"].head(5), "\n")

        processing_time = time.time() - start_time  #calculate duration 
        logger.info(f"Pure feature engineering and data processing in: {processing_time:.2f} seconds\n")
    
    except Exception as e:
        logger.error(f"Error in Data preparing/feature engineering: {e}")
        return None
    

    try:
        third_class = visualizations(data)
        third_class.plot()
        
    except Exception as e:
        logger.error(f"Error in Data visualization: {e}")
        return None

    """

    #### testing ETL pipeline:
    file_paths = {
        'movies': 'movies.dat',
        'ratings': 'ratings.dat',
        'tags': 'tags.dat',
        'movie_ratings': 'Movie_ratings.csv' 
    }
    extractor = moviedataextractor(chunk_size=10000)
    transformer = moviedatatransform()
    loader = moviedataload("data/processed")
    """
    try: 
        #extract
        raw_data = extractor.extract_from_source(file_paths)
        
        #transform
        transformed_data, genres, stats = transformer.transform(raw_data)

        #load
        loader.load_data(transformed_data, partition_cols=['year'])

        processing_time = time.time() - start_time  #calculate duration 
        logger.info(f"Total processed time is: {processing_time:.2f} seconds")

        return transformed_data
    
    except Exception as e:
        logger.error(f"Pipline failed: {e}")
        return None
    """

    #lets test the matrix factorization with our raw data now:

    """
    try:
    
        # Train model
        logging.info("Starting model Training")
        try:
            model, ratings_scale, (user_mapping, movie_mapping) = matrix_fact.train_model(
                data["movie_ratings"],
                n_factors=30,
                n_epochs=100
            )
            logging.info("Model training completed Successfully!")
            logging.info(f"Created embeddings for {len(user_mapping)} users and {len(movie_mapping)} movies")
        
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.error(f"GPU out of memory error: {e}")
                # Could add CPU fallback here
            else:
                raise e
            
        processing_time = time.time() - start_time  #calculate duration 
        logger.info(f"Model Training in:  {processing_time:.2f} seconds\n")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        return None
    """
    #Lets train the model on the ETL we saved!
    try:

        # 1. Load processed data
        logger.info("Initializing data loader...")
        base_path = Path("data/processed")

        logger.info(f"Loading data from parquet files")
        try:
            # Read partitioned parquet data
            movie_ratings = pd.read_parquet(str(base_path / "movie_ratings"),engine='pyarrow')
            
            logger.info(f"Original Dataset size: {len(movie_ratings):,}")

            #take a sample for training the model quicker:
            movie_ratings = movie_ratings.sample(n=750000, random_state= 42)
            logger.info(f"Sampled dataset size: {len(movie_ratings):,}")
            
            logger.info(f"Unique users in sample: {movie_ratings['userID'].nunique():,}")
            logger.info(f"Unique movies in sample: {movie_ratings['movieID'].nunique():,}")
            logger.info(f"Memory usage: {movie_ratings.memory_usage().sum() / 1024**2:.2f} MB")
            
            
            # 3. Train model with optimized parameters
            logger.info("Starting model training...")
            model, ratings_scale, (user_mapping, movie_mapping) = matrix_fact.train_model(
                movie_ratings,
                n_factors=128,
                n_epochs=50, 
                batch_size=4096
            )

            #save model/mappings
            save_path = Path("model")
            save_path.mkdir(exist_ok= True)

            #save model state:
            torch.save(model.state_dict(), save_path / "movie_recommender_model.pth")

            #save these mappings for future use:
            mapping_info = {
                'user_mapping': user_mapping,
                'movie_mapping': movie_mapping,
                'ratings_scale': ratings_scale
            }
            torch.save(mapping_info, save_path / "mapping_info.pth")

            logger.info("model training completed successfully!!")
            #logger.info(f"Created embeddings for {len(user_mapping):,} users and {len(movie_mapping):,} movies")
            #logger.info(f"Model and mappings saved to {save_path}")

             #lets test the model now:
    
            logger.info("Testing the model predictions...")

            # Get the first few users for testing
            test_users = movie_ratings['userID'].unique()[:5]  # Test first 5 users
            logger.info(f"Testing predictions for users: {test_users}")

            with torch.no_grad():
                for test_user in test_users:
                    user_data = movie_ratings[movie_ratings['userID'] == test_user].head()  # Get first 5 movies for each user
                    logger.info(f"\nPredictions for user {test_user}:")
                    
                    if len(user_data) > 0:
                        for _, row in user_data.iterrows():

                            # Calculate movie age for prediction
                            rating_year = pd.to_datetime(row['timestamp']).year
                            release_year = float(re.search(r'\((\d{4})\)', row['title']).group(1))
                            movie_age = rating_year - release_year
                            age_bucket = int(np.floor(max(0, movie_age) / 5))

                            #convert to tensors
                            user = torch.LongTensor([user_mapping[row['userID']]]).to(DEVICE)
                            movie = torch.LongTensor([movie_mapping[row['movieID']]]).to(DEVICE)
                            age = torch.LongTensor([age_bucket]).to(DEVICE)

                            #get predictions
                            prediction = model(user, movie, age)

                            scaled_prediction = prediction.item() * ratings_scale
                            rounded_prediction = round(scaled_prediction *2)/2  #Round to the nearst 0.5

                            #Log with additional information
                            logger.info(f"Movie {row['movieID']} ({row['title']}): "
                            f"Predicted = {rounded_prediction:.1f}, "  # .1f for consistent decimal places
                            f"Actual = {row['rating']} "
                            f"(Movie age when rated: {movie_age:.1f} years)")
                    else:
                        logger.info(f"No ratings found for user {test_user}")

        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.error("GPU memory error. Consider: ")
                logger.error(" Reducing batch_size, Reducing n_factors or taking a sample from the data (50,000)")
                raise e
        
        total_time = time.time() - start_time
        logger.info(f" Total Processing time: {total_time:.2f} seconds")

        return model, mapping_info
    
    except Exception as e:
        logger.error(f"Error in training model on pipeline: {str(e)}")
        return None



if __name__ == '__main__':
    final_result = main()
   




