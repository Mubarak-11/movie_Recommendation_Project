## The goal of this file is to take the ETL, prepare it for the model definition ##

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy
import logging, os
from etl_pipeline import load
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class book_reco_Dataset(Dataset):
    """ Handle how indiviual samples are accessed, taking advantage of pytorch Dataset"""
    def __init__(self, books_dataset):

        df = books_dataset.copy()

        #handle all numeric columsn to ensure clean tensor conversion:
        numeric_columns = ['UserID', 'BookID', 'Interaction', 'Binary_ratings']
        

        for col in numeric_columns:
            if col in df.columns:
                #convert to numeric, forcefully:
                df[col] = pd.to_numeric(df[col], errors = 'coerce')

                #fill NaN values with 0
                df[col] = df[col].fillna(0)

                #ensure values are finite (replace infinite values with 0)
                df[col] = df[col].replace([np.inf, -np.inf], 0)

                #finally convert to int safely
                df[col] = df[col].astype(int)
        
        # If you need to create Interaction from BookRating, do it safely
        if 'BookRating' in df.columns:
            df['Interaction'] = df['BookRating'].fillna(0)
        else:
            df['Interaction'] = df['Interaction'].fillna(0)
        
        #convert integers to avoid potential float -> long conversion issues
        self.UserID = torch.LongTensor(df['UserID'].values)
        self.BookID = torch.LongTensor(df['BookID'].values)
        self.Interaction = torch.LongTensor(df['Interaction'].values)
        self.Binary_ratings = torch.LongTensor(df['Binary_ratings'].values)
        

        #additional features
        self.book_popularity = torch.FloatTensor(books_dataset['book_popularity'].values)
        self.author_popularity = torch.FloatTensor(books_dataset['author_popularity'].values)
        self.publisher_popularity = torch.FloatTensor(books_dataset['publisher_popularity'].values)
        self.user_activity = torch.FloatTensor(books_dataset['user_activity'].values)

    def __len__(self):
        return len(self.UserID) #return the number of samples we have (UserID or BookID, both have the same length)

    def __getitem__(self, index):

        return {
            'UserID': self.UserID[index],
            'BookID': self.BookID[index],
            'Interaction': self.Interaction[index],
            'Binary_ratings': self.Binary_ratings[index],

            #additional features
            'book_popularity': self.book_popularity[index],
            'author_popularity': self.author_popularity[index],
            'publisher_popularity': self.publisher_popularity[index],
            'user_activity': self.user_activity[index]
        }


class data_prepare_ml():
    """ data preparation functions."""

    def __init__(self):
        
        #intilize empty mappings to use below
        self.user_map = {}
        self.bookid_map = {}
   

    def fit(self, books_dataset):
        """Fit mappings from categorical values to unique integer indices."""

        #create mapping for userIDs
        self.user_map = {old_id: new_id for new_id, old_id in enumerate(books_dataset['UserID'].unique())}
        self.bookid_map = {old_id: new_id for new_id, old_id in enumerate(books_dataset['BookID'].unique())}
        
        #Features Added:
        
    def transform(self, books_dataset):

        """Apply fitted mappings to transform categorical columns to integers."""
        df = books_dataset.copy()

        #apply the mappings now
        df['UserID'] = df['UserID'].map(self.user_map)
        df['BookID'] = df['BookID'].map(self.bookid_map)
        
        #applying mapping on features
        

        return df
    
    def create_non_leaking_features(self, books_dataset):
        """
        Adding popularity features in a non-leaking way
        """

        logger.info("Creating non-leaking popularity features")

        df = books_dataset.copy()

        #first feature: book popularity (count-based)
        book_counts = df.groupby('BookTitle').size()
        df['book_popularity'] = df['BookTitle'].map(book_counts).fillna(0)

        #second feature: Author popularity
        author_counts = df.groupby('Author').size()
        df['author_popularity'] = df['Author'].map(author_counts).fillna(0)

        #third feature: publisher popularity
        publisher_counts = df.groupby('Publisher').size()
        df['publisher_popularity'] = df['Publisher'].map(publisher_counts).fillna(0)

        #fourth feature: user activity
        user_counts = df.groupby('UserID').size()
        df['user_activity'] = df['UserID'].map(user_counts).fillna(0)

        #Normalize all features to [0,1] range for BCF
        for feature in ['book_popularity', 'author_popularity', 'publisher_popularity', 'user_activity']:
            max_val = df[feature].max()
            if max_val > 0:
                df[feature] = df[feature]/ max_val
        
        logger.info('Non-leaking features created successfully!')

        return df
    
    def get_vocab_sizes(self):
        """
        Return the number of unique entries for each categorical feature for embeddings.
        """
        return{
            'n_userID': len(self.user_map),
            'n_bookID': len(self.bookid_map),
        }
    

    def analyze_ratings(self, books_dataset):
        """ Code for Analyzing and visiulzing bookrating/interaction distribtuion"""

        plt.figure(1)
        books_dataset['Interaction'] = books_dataset['BookRating'].astype(float)
        plt.hist(books_dataset['Interaction'], bins = 20, color= 'blue', edgecolor = 'black')
        plt.title("distrubtion of Interactions")
        plt.xlabel("Rating")
        plt.ylabel("Frequency")
        
        """ So from this plot:  we determine that the threshold for binary conversion should be
            Ratings ≥ 7 are positive interactions (1), Ratings < 7 are negative interactions (0)"""

        
    def calc_sparsity(self, books_dataset):
        """Calculate matrix sparsity, formula is spars = # of zero elements/total # of elements"""
        
        zero_values = (books_dataset['Interaction'] == 0).sum()
        total_values = books_dataset['Interaction'].shape[0] #shape will give us an integer of the total number of rows

        sparsity = (zero_values/total_values)*100
        logger.info(f"Sparsity for book interactions is: {sparsity:.2f}")

        #will also add interaction density to get the full picture:
        density = 100 - sparsity
        logger.info(f"Density for book interaction is: {density}")


    def conv_binary(self, books_dataset):
        """Convert ratings/interactions to binary implicit feedback
            Ratings ≥ 7 are positive interactions (1), Ratings < 7 are negative interactions (0)"""

        books_dataset['Binary_ratings'] = books_dataset['Interaction'].apply(
            lambda x: 1 if x>= 7 else 0
        )
        
        logger.info(f"Binary distribution: {books_dataset['Binary_ratings'].value_counts()}")
        logger.info(f"Empty rows in Binary columns: {books_dataset['Binary_ratings'].isnull().sum()}")

    def negative_sampling(self, books_dataset, neg_ratio = 4): #4 or 8 for neg ratio
        """Generate custom negative samples for training"""
        
        #defining the popularity metric
        books_dataset['Interaction_count'] = books_dataset.groupby('BookID')['BookID'].transform('count')

        #generate unique users/books
        uniq_userID = books_dataset['UserID'].unique()
        uniq_bookID = books_dataset['BookID'].unique()

        #positive (user, book) pairs
        positive_pairs = set(zip(
            books_dataset[books_dataset['Binary_ratings'] == 1]['UserID'],
            books_dataset[books_dataset['Binary_ratings'] == 1]['BookID']
        ))

        #create a dictionaray mapping each book to its popularity score
        book_pop = {book: books_dataset[books_dataset['BookID'] == book]['Interaction_count'].values[0]
                           if any (books_dataset['BookID'] == book) else 1
                           for book in uniq_bookID}

        #convert that dict to a numpy array that matches the order of uniqBookID
        popularity_weights = np.array([book_pop[book] for book in uniq_bookID])

        #normalize weights to create proper probability distribution (sums up 1)
        popularity_weights = popularity_weights/ popularity_weights.sum()


        negative_samples = []

        #for each user, sample negative books
        for user in uniq_userID:
            #find books this user has rated positivley
            user_positive_books = set(books_dataset[(books_dataset['UserID'] == user) &
                                        (books_dataset['Binary_ratings'] == 1)]['BookID'])

            #calculate number of negative samples needed for this user
            num_negatives = len(user_positive_books) * neg_ratio
            
            #skip users with no positive interactions
            if num_negatives == 0:
                continue

            #set to collect negative samples for this user
            user_negatives = set()

            #keep sampling until we have enough negative samples
            while len(user_negatives) < num_negatives:

                sampled_books = np.random.choice(
                    uniq_bookID,
                    size = num_negatives * 2, #oversample than trim
                    p = popularity_weights
                )

                #filter out books the user has already positively rated
                for book in sampled_books:
                    if book not in user_positive_books and (user,book) not in positive_pairs:
                        user_negatives.add((user, book))
                    if len(user_negatives) >= num_negatives:
                        break
            
            #convert each negative sample to dict format for dataframe
            for user_book in user_negatives:
                negative_samples.append({'UserID' : user_book[0], 'BookID': user_book[1], 'Binary_ratings':0})
        
        #build negative dataframe
        negative_df = pd.DataFrame(negative_samples)

        #filter only positive samples
        positive_df = books_dataset[books_dataset['Binary_ratings'] == 1]

        #combine
        combined_samples = pd.concat([positive_df, negative_df], ignore_index=True)

        # Log statistics about the sampling process
        logger.info(f"Total samples: {len(combined_samples)}")
        logger.info(f"Positive samples: {len(positive_df)}")
        logger.info(f"Negative samples: {len(negative_df)}")
        logger.info(f"Negative-to-positive ratio: {len(negative_df)/len(positive_df):.2f}")

        return combined_samples
       

def main():
    #Main function to test everything
    
    try:
        #Lets test out loading the ETL
        books_dataset = load.read_data("Books_dataset.csv", sep = ",")
        logger.info(f" We have successfully loaded the books dataset, the size is: {len(books_dataset)}")
        
    except Exception as e:
        logger.error(f" failed to Load ETL: {str(e)}")

    try:

        #data prepartion off to dataset class
        data_prep = data_prepare_ml()

        #run pre-processing steps
        data_prep.fit(books_dataset)    #fit the dataset
        book_mapped = data_prep.transform(books_dataset) #transform and apply the mappings
        
        data_prep.analyze_ratings(books_dataset)
        data_prep.calc_sparsity(books_dataset)
        data_prep.conv_binary(books_dataset)
        
        #apply negative sampling
        combined_samples = data_prep.negative_sampling(books_dataset)
        
        #create and save the prepare dataset
        os.makedirs("prepared_data", exist_ok=True)

        combined_samples = data_prep.create_non_leaking_features(combined_samples)
        # Then save the file
        combined_samples.to_parquet(
            "prepared_data/train_with_features.parquet", 
            index=False, 
            engine='pyarrow', 
            compression='snappy'
        )
        
        #train_dataset = book_reco_Dataset(books_dataset)
    except Exception as e:
        logger.error(f"Failed during data preparation Stage due to: {str(e)}")


        
if __name__ == '__main__':
    main()
