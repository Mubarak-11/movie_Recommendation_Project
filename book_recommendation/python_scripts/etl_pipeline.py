### This python file Extracts the data from dataset folder, transforms and loads a clean/transformed data in csv format

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging, time

logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class extract():
    """ Extract the Dataset and Combine them into a single csv"""
    def __init__(self):
        pass
    
    def grab_csv(filepath)-> pd.DataFrame:
        """ This function extracts the dataset, combines the csv files and prepares for transformation"""

        read_data = pd.read_csv(filepath, encoding = "latin1", sep = ",", dtype="object")

        data = pd.DataFrame(read_data)

        logger.info(f"Finished reading data from {filepath}")
        return data

    def combine_csv(df1: pd.DataFrame, df2: pd.DataFrame, df3: pd.DataFrame) -> pd.DataFrame:
        """ This function will take all 3 dataframes and combine into one dataframe"""

        chunk_size = 100000
        merged_chunks = []
        
        for merge1 in range(0, len(df1), chunk_size):
            chunk1 = df1.iloc[merge1: merge1 + chunk_size]  #get a chunk from df1

            book_ratings = pd.merge(chunk1, df2, on= "ISBN", how="inner")    # combine books + ratings on common column (ISBN)
            #print("Merging books and ratings on ISBN")
            
            for merge2 in range(0, len(df3), chunk_size):
                chunk2 = df3.iloc[merge2: merge2 + chunk_size] #get a chunk from df3

                books_ratings_users = pd.merge(book_ratings, chunk2, on="User-ID", how="inner")  # combine book_ratings + Users on common column (User-ID)
                #print("Merging book_ratings and users on User-ID")

            merged_chunks.append(books_ratings_users)   #store merged chunks

        #combine all merged chunks
        combined_df = pd.concat(merged_chunks, ignore_index= True)
        
        
        #combined_df.to_csv("combined_output.csv")
        logger.info("Merged chunks and save as a csv file ")

        return combined_df  #return combined df
        
class transform():
    """ Handle data transformation and cleaning """
    def __init__(self):
        pass

    def cleaning(df: pd.DataFrame) -> pd.DataFrame:
        """ rename columns, remove unnecessary columns"""

        df = df.drop(columns = ["Image-URL-S", "Image-URL-M", "Image-URL-L", ])   #drop these columns
        df = df.rename(columns={"Unnamed: 0": "Index", "ISBN": "BookID", "Book-Title": "BookTitle", "Book-Author": "Author",    #Rename columns
                                "Year-Of-Publication": "Yearpublish", "User-ID": "UserID", "Book-Rating": "BookRating"})
         
        return df

    def post_clean_transform(df: pd.DataFrame) -> pd.DataFrame:
        """ start transformations after pre-cleaning"""

        #lets deal with the missing rows in the Age column using Random Imputation
        actual_ages = df['Age'] = df['Age'].dropna() #drop the missing entries first

        missing_mask = df['Age'].isna()
        df.loc[missing_mask, 'Age'] = np.random.choice(
            actual_ages, 
            size=missing_mask.sum(),
            replace = True          #fill that column now
        )

        logger.info("Cleaned all missing values in age")

        #convert locations(city, state, country) into just countries
        #df['Location'] = df['Location'].apply(lambda x: x.split(',')[2])
        #df['Location'] = df['Location'].apply(lambda x: x.upper())

        #convert BookRating to numeric, its currently oan object
        df['BookRating'] = pd.to_numeric(df['BookRating'], errors= "coerce")

        #categorical encoding of string columns! 
        Book_per_rating = df.groupby('BookTitle')['BookRating'].mean()  #encode for a new column
        df['Book_per_rating_encoded'] = df['BookTitle'].map(Book_per_rating) 

        author_per_rating = df.groupby('Author')['BookRating'].mean()
        df['Author_per_rating_encoded'] = df['Author'].map(author_per_rating) 

        publisher_per_rating = df.groupby('Publisher')['BookRating'].mean()
        df['Publisher_per_rating'] = df['Publisher'].map(publisher_per_rating)
        
        logger.info("New columns created using their means/medians")

        #convert Age to numeric, its currently oan object
        df['Age'] = pd.to_numeric(df['Age'], errors= "coerce")
        
        median_age_country = df.groupby("Location")['Age'].median()
        df['Age'] = df.apply(lambda row: median_age_country.get(row['Location'])
                             if pd.isna(row['Age']) else row['Age'], axis=1)

        df['explicit_rating'] = df['BookRating'] > 0 #new column to distingush  between explicit ratings

        return df
    
class load():
    """ Load the extracted + transformed into parquet/csv format"""
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def load_data(df: pd.DataFrame):
        """ Take the extraced/transformed data and save into a csv/parquet file """
        
        df.to_csv('Books_dataset.csv', index = False)
        logger.info("Data saved to the Books_dataset.csv")

    @staticmethod
    def read_data(filepath: str, sep: str = ",")->pd.DataFrame:
        """ Read that data we just loaded above, read from a csv file"""
        
        df = pd.read_csv(filepath, sep=",", dtype="object")
        logger.info(f"Data Loaded from {filepath}")

        return df

#lets test this
def main():
    """ Main function to load/process data files"""

    start_time = time.time()
    file_paths = {
        "books": "/home/bruno/Documents/book_recommendation/dataset/raw/Books.csv",
        "ratings": "/home/bruno/Documents/book_recommendation/dataset/raw/Ratings.csv",
        "users": "/home/bruno/Documents/book_recommendation/dataset/raw/Users.csv"
    }

    try:
        dataframes = []
        #load all 3 csv files
        for name, path in file_paths.items():
            data = extract.grab_csv(path)
            dataframes.append(data)

        processing_time = time.time() - start_time
        logger.info(f" Time duration for extract stage is : {processing_time:.2f} seconds")

         # grab csv + combine csv files into a single Dataframe
        combined_df = extract.combine_csv(*dataframes)  # *unpack

        # Save combined DataFrame to CSV
        #combined_df.to_csv("/home/bruno/Documents/book_recommendation/python_scripts/combined_output.csv", index=False)

        # Read the combined CSV file
        to_df = load.read_data("/home/bruno/Documents/book_recommendation/python_scripts/combined_output.csv", sep=",")
        
        
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
    
    
    try:
        logger.info(f"How many columns before dropping one: {len(to_df.columns)}")
        cleaned_df = transform.cleaning(to_df)
        logger.info(f"How many columns after dropping one: {len(cleaned_df.columns)}")
        logger.info(f"\nRemaining columns: {cleaned_df.columns}")
        
        logger.info(f"\n which columns have Missing data: {cleaned_df.isnull().sum()}")

        #post_cleaning_transform
        cleaned_df = transform.post_clean_transform(cleaned_df)
        
        logger.info(f"\n We successfully random imputated missing data: {cleaned_df.isnull().sum()} \n")
    
        logger.info(f" Printing the new columns: {cleaned_df[['Book_per_rating_encoded','Author_per_rating_encoded','Publisher_per_rating']].iloc[0:5]}")

        processing_time = time.time() - start_time
        logger.info(f" Time duration for transform stage is: {processing_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Failures in post_clean transform: {str(e)}")
    
    #Load pipeline
    load.load_data(cleaned_df)

    logger.info("ETL process is completed and saved! ")

    processing_time = time.time() - start_time
    logger.info(f" Total Processing time is: {processing_time:.2f} seconds")

if __name__ == '__main__':
    main()