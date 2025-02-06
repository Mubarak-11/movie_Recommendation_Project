### Building a movie_recommendation system, Clean and concise, trying to write efficient and easy code

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import gc,time,psutil,functools
from typing import Iterator, Dict
import logging

logging.basicConfig(level = logging.INFO, format='%(asctime)s- %(levelname)s - %(message)s')
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

        plt.show()

#create a seperate class for implementing different recommendtion alogrithms
#create a seperate class for ETL 


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
        data['movies'] = processor.process_file('movies.dat', sep='::')
        data['ratings'] = processor.process_file('ratings.dat', sep='::')
        data['tags'] = processor.process_file('tags.dat', sep='::')
        
        # Load the main CSV file
        logger.info("Processing main CSV file...")
        data['movie_ratings'] = processor.process_file('Movie_ratings.csv', sep=',')
        #movie_ratings = processor.process_file('Movie_ratings.csv', sep=',')
        print(data['movie_ratings'].head(),"\n")

        processing_time = time.time() - start_time  #calculate duration 
        logger.info(f"Data loading in:  {processing_time:.2f} seconds\n")
        
    except Exception as e:
        logger.error(f"Error in Data Loading: {e}")
        return None
    
    try:
        #pass the data to the second class: 
        second_class = feature_eng_and_data_clean(data)

        # feature engineering #1
        data = second_class.feature_eng_1() #store the processed data
        print("The year column contains: ",data["movie_ratings"]["year"].isnull().sum(), " empty rows\n")
        
        #feature engineering #2
        genres_encoded = second_class.feature_eng_2()
        data["movie_ratings"] = pd.concat([data["movie_ratings"], genres_encoded], axis=1) 
        #print(data["movie_ratings"].head())
        
        #feature engineering #3
        rating_stats = second_class.feature_eng_3()
        print("printing the stats :", rating_stats.head(),"\n")

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


    finally:
        plt.close('all') #clean and close all plots!
      
      
    processing_time = time.time() - start_time  #calculate duration 
    logger.info(f"Total processed time is: {processing_time:.2f} seconds")
   
    return data # Return data after all operations are complete

if __name__ == '__main__':
    final_result = main()




