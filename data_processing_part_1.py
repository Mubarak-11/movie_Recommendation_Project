### data_processing.py - Handles data loading, feature engineering, and ETL processes

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import gc, time, logging
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Dict, List, Optional

# Basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s- %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

class FeatureEngAndDataClean:
    """Class for feature engineering and data cleaning"""
    def __init__(self, data):
        self.data = data
    
    # Drop missing rows and feature_engineering 1: extract year from movie title
    def feature_eng_1(self):
        if "tags" in self.data:
            self.data["tags"] = self.data["tags"][self.data["tags"].notna()]  # drop missing rows
        self.data['movie_ratings']['year'] = self.data['movie_ratings']['title'].str.extract(r"\((\d{4})\)")
        return self.data
    
    # Feature_engineering 2: genre column encoding
    def feature_eng_2(self):
        self.genres_encoded = self.data["movie_ratings"]["genres"].str.get_dummies("|")
        return self.genres_encoded
    
    # Feature engineering 3: Rating statistics
    def feature_eng_3(self):
        self.rating_stats = self.data["movie_ratings"].groupby(["title"]).agg({
             "rating": ["size", "mean", "median", "std"] # size is the number of ratings
        })
        return self.rating_stats

    # Feature engineering 4: Extract Timestamps features, converting timestamp column to datetime format
    def feature_eng_4(self):
        self.data["movie_ratings"]["timestamp"] = pd.DatetimeIndex(
            pd.to_datetime(self.data["movie_ratings"]["timestamp"],
                            unit="s", 
                            cache=True
            )
        )
        return self.data

class Visualizations:
    """Class to utilize the features and plot them with matplotlib/seaborn"""
    def __init__(self, data):
        self.data = data
        self.rating_stats = self.data["movie_ratings"].groupby(["title"]).agg({
             "rating": ["size", "mean", "median", "std"] # size is the number of ratings
        })

    def plot(self):
        # Plot 1: top 10 most rated movies
        plt.figure(figsize=(10, 6))
        top_movies = self.rating_stats["rating"]["size"].nlargest(10)
        sns.barplot(x=top_movies.values, y=top_movies.index, color="red")
        plt.title("Top 10 Most Rated Movies")
        plt.xlabel("Number of Ratings")
        plt.tight_layout()

        # Plot 2: User activity vs Average Rating
        plt.figure(figsize=(10, 6))
        plt.scatter(self.rating_stats["rating"]["size"], self.rating_stats["rating"]["mean"], 
                   edgecolors="k", color="purple", alpha=0.5)
        plt.title("User Activity vs Average Rating")
        plt.xlabel("Number of Ratings")
        plt.ylabel("Average Rating")
        plt.grid(True)
        plt.tight_layout()

        # Plot 3: Distribution of Ratings
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.data["movie_ratings"], x="rating", bins=10, kde=True, color="green")
        plt.title("Distributions of Ratings")
        plt.ylabel("Rating")
        plt.xlabel("Count")
        plt.tight_layout()

        # Plot 4: Correlation matrix
        plt.figure(figsize=(10, 8))
        numeric_columns = self.data["movie_ratings"].select_dtypes(include=[np.number])
        self.correlation_matrix = numeric_columns.corr()
        sns.heatmap(self.correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.title("Heatmap of Correlation between User-Movie Features")
        plt.tight_layout()

        plt.show()

# ETL Pipeline Classes
class MovieDataExtractor:
    """Handles data source connections and raw data extraction"""
    def __init__(self, chunk_size: int = 500000):
        self.chunk_size = chunk_size
        self.logger = logging.getLogger(__name__)

    def validate_data(self, data):
        """Validate data structure/content before extraction"""
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
        """Extract data from source files"""
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

class MovieDataTransform:
    """Handles data transformation and cleaning"""
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.quality_metrics = {}

    def check_data_quality(self, data, stage="pre"):
        """Check data quality before/after transformation"""
        metrics = {
            'missing_values': data.isnull().sum().to_dict(),
            'unique_counts': {col: data[col].nunique() for col in data.columns},
            'memory_usage': data.memory_usage(deep=True).sum() / 1024**2  # MB
        }
        self.quality_metrics[f"{stage}_transform"] = metrics
        return metrics

    def transform(self, data):
        """Applies all transformation steps to the data"""
        start_time = time.time()
        try:
            # Pre-transform quality check
            self.check_data_quality(data["movie_ratings"], "pre")

            transformed_data = {
                'tags': data['tags'][data['tags'].notna()] if 'tags' in data else None,
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
                    transformed_data["movie_ratings"]["timestamp"],
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

class MovieDataLoad:
    """Handles data storage and retrieval"""
    def __init__(self, storage_path):
        self.storage_path = Path(storage_path)
        self.logger = logging.getLogger(__name__)
        self.storage_path.mkdir(parents=True, exist_ok=True)
    
    def load_data(self, data, partition_cols=None):
        """Load data into parquet format with optional partitioning"""
        start_time = time.time()
        try:
            for key, df in data.items():
                if df is None:
                    continue
                    
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
                        use_dictionary=True
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
            self.logger.error(f"Loading failed: {str(e)}")
            raise

    def read_data(self, dataset_name):
        """Reads data from parquet storage"""
        try:
            filepath = self.storage_path / f"{dataset_name}"
            return pd.read_parquet(filepath)
        except Exception as e:
            self.logger.error(f"Reading failed: {str(e)}")
            raise

def main():
    """
    Main function to load and process data files.
    Returns a dictionary containing all processed DataFrames.
    """
    start_time = time.time()
    processor = DataProcess(chunk_size=10000)
    data = {}  # Dictionary to store all processed data
    
    try:
        # Load the movie ratings data
        logger.info("Processing CSV files...")
        data['movie_ratings'] = processor.process_file('Movie_ratings.csv', sep=',')
        
        logger.info(f"Original dataset size: {len(data['movie_ratings'])}")
        
        # Sample for faster processing during testing
        data['movie_ratings'] = data['movie_ratings'].sample(n=50000, random_state=42)
        logger.info(f"Sampled dataset size: {len(data['movie_ratings'])}")
        
        # Debug info for user and movie IDs
        n_users = data['movie_ratings']['userID'].nunique()
        n_movies = data['movie_ratings']['movieID'].nunique()
        
        logger.info(f"Number of unique users: {n_users}")
        logger.info(f"Number of unique movies: {n_movies}")
        
        # Apply feature engineering
        feature_eng = FeatureEngAndDataClean(data)
        
        # Feature engineering steps
        data = feature_eng.feature_eng_1()
        genres_encoded = feature_eng.feature_eng_2()
        data["movie_ratings"] = pd.concat([data["movie_ratings"], genres_encoded], axis=1)
        rating_stats = feature_eng.feature_eng_3()
        data = feature_eng.feature_eng_4()
        
        # Optional visualization
        # visualizer = Visualizations(data)
        # visualizer.plot()
        
        # ETL Pipeline
        storage_path = Path("data/processed")
        storage_path.mkdir(parents=True, exist_ok=True)
        
        # Load the transformed data into parquet storage
        loader = MovieDataLoad(storage_path)
        loader.load_data(data, partition_cols=['year'])
        
        processing_time = time.time() - start_time
        logger.info(f"Total processing time: {processing_time:.2f} seconds")
        
        return data
        
    except Exception as e:
        logger.error(f"Error in data processing: {e}")
        return None

if __name__ == '__main__':
    processed_data = main()