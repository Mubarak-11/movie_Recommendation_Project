# Goal of this file is to take prepare_model.py and build the NeuMF model

from prepare_model import book_reco_Dataset
from prepare_model import data_prepare_ml
import pandas as pd, numpy as np
import logging, os, time
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterSampler


logging.basicConfig(level = logging.INFO, format= '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
logger.info(f" Using: {device}")

class model_def(nn.Module):
    """ NeuMF model implementation
            - Normal MF (for linear interactions)
            - MLP (multi layer perceptron (non-linaer interactions) 
    """

    def __init__(self, n_userID, n_BookID, 
                 #num_book_rating_bins, num_author_bins, num_publisher_bins, num_year_bins, num_age_bins ,
                  mlp_size = [256,128, 64, 32], n_factors = 100):
        super().__init__()

        self.logger = logging.getLogger(__name__)

        #create embeddings for MF
        self.mf_user_factors = nn.Embedding(n_userID, n_factors)
        self.mf_book_factors = nn.Embedding(n_BookID, n_factors)

        #embeddings for MLP
        self.mlp_user_embeddings = nn.Embedding(n_userID, mlp_size[0] // 2)
        self.mlp_book_embeddings = nn.Embedding(n_BookID, mlp_size[0] // 2)
      
        #calculate total MLP input size
        base_embed_size = mlp_size[0] # 256 (128+128)
        num_continuous_features = 4  # 4 continous features 
        mlp_input_dim = base_embed_size + num_continuous_features 

        #MLP layers
        self.mlp_layers = nn.ModuleList()
        layer_dims = [mlp_input_dim] + mlp_size[1:] #e.g [260, 128, 64, 32]
        for i in range(len(mlp_size) -1):   
            self.mlp_layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            self.mlp_layers.append(nn.SiLU())
            self.mlp_layers.append(nn.Dropout( p = 0.2))

        #final prediction layer (MF + MLP)
        self.output_layer = nn.Linear(n_factors + mlp_size[-1], 1)
        nn.init.xavier_normal_(self.output_layer.weight) #normalize here too for consistency!  
        self.output_layer.bias.data.fill_(0.)

        #initilize the weights (mf/mlp) using xavier/Glorot normal initializations
        for embed in [
            self.mf_user_factors, 
            self.mf_book_factors,
            self.mlp_user_embeddings,
            self.mlp_book_embeddings,
        ]:
            nn.init.xavier_normal_(embed.weight)
        
        self.to(device)

    def forward(self, user_ids, book_ids, continuous_features
                #book_rating_ids, author_ids, publish_ids, year_id, age_val
                ):
        """
        Forward pass to predict ratings
        """
        users_mf = self.mf_user_factors(user_ids)
        books_mf = self.mf_book_factors(book_ids)


        #element wise operation
        mf_vector = torch.mul(users_mf, books_mf)

        #mlp path
        users_mlp = self.mlp_user_embeddings(user_ids)
        books_mlp = self.mlp_book_embeddings(book_ids)

        mlp_vector = torch.cat([users_mlp,
                                books_mlp,
                                continuous_features
                                ], dim= 1)  #concatenates users/books with features and passes them through multiple Neural layers

        #process through MLP layers
        for layer in self.mlp_layers:
            mlp_vector = layer(mlp_vector)

        #concatenate MF + MLP paths
        predict_vector = torch.cat([mf_vector, mlp_vector], dim=1) 

        #make predictions
        logits = self.output_layer(predict_vector)

        return logits.squeeze()

    @staticmethod
    def train_model(train_data, val_data, vocab_sizes, learning_rate, weight_decay, n_factors = 50, n_epochs = 50, batch_size = 1024):
        """ 
        Train the NeuMF model, with validation
        """
        n_userID = vocab_sizes['n_userID']
        n_BookID = vocab_sizes['n_bookID']
  

        model = model_def(n_userID,
                          n_BookID,
                          mlp_size=[256, 128, 64, 32], n_factors=n_factors)

        optimizer = optim.Adam(model.parameters(), lr = learning_rate , weight_decay = weight_decay)

        #loss function
        criteron = nn.BCEWithLogitsLoss()

        #create train dataloader
        train_loader = DataLoader(book_reco_Dataset(train_data), batch_size= batch_size, shuffle= True)
        
        #create validation dataloader
        val_loader = DataLoader(book_reco_Dataset(val_data), batch_size = batch_size, shuffle= False)

        # keep track of best model and track best validation loss
        best_model_state = None
        best_val_loss = float('inf')

        #lets train now
        for epochs in range(n_epochs):

            #training phase
            model.train()
            total_train_loss = 0
            train_count = 0

            for batch in train_loader:
                users = batch['UserID'].to(device)
                books = batch['BookID'].to(device)
                target = batch['Binary_ratings'].float().to(device)

                #Features added
                continous_features = torch.cat([
                    batch['book_popularity'].unsqueeze(1).to(device),
                    batch['author_popularity'].unsqueeze(1).to(device),
                    batch['publisher_popularity'].unsqueeze(1).to(device),
                    batch['user_activity'].unsqueeze(1).to(device),
                ], dim = 1)


                #clear previous gradients
                optimizer.zero_grad()

                #forward pass
                predictions = model(users, books, continous_features)

                #calculate loss
                loss = criteron(predictions, target)

                #backward propagation
                loss.backward()

                #update parameters
                optimizer.step()

                total_train_loss += loss.detach().item()
                train_count += 1
        
            #validation phase
            model.eval()
            total_val_loss = 0
            val_count = 0

            with torch.no_grad(): #no need to track gradients during validation
                for batch in val_loader:
                    users = batch['UserID'].to(device)
                    books = batch['BookID'].to(device)
                    target = batch['Binary_ratings'].float().to(device)

                    #Features added
                    continous_features = torch.cat([
                        batch['book_popularity'].unsqueeze(1).to(device),
                        batch['author_popularity'].unsqueeze(1).to(device),
                        batch['publisher_popularity'].unsqueeze(1).to(device),
                        batch['user_activity'].unsqueeze(1).to(device),
                    ], dim = 1)

                    predictions = model(users, books, continous_features)
                    loss = criteron(predictions, target)

                    total_val_loss += loss.item()
                    val_count += 1
            
            train_loss = total_train_loss / train_count
            val_loss = total_val_loss / val_count

            logging.info(f" Epoch {epochs + 1}/{n_epochs} Train loss: {train_loss:.4f}, Validation loss: {val_loss:.4f} ")

                #save the best model, if validation improves
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                logging.info(f"New best Model saved with validation loss: {best_val_loss:.4f}")
                
        #load the best model state
        model.load_state_dict(best_model_state)

        return model

    @staticmethod
    def calc_valid(model, eval_dataset, batch_size = 1024):
        """
        Calculate and return validation loss on the model given the dataset
        """

        #create dataset/dataloader
        eval_data = book_reco_Dataset(eval_dataset)
        eval_loader = DataLoader(eval_data, batch_size= batch_size, shuffle= False)

        #setup loss function
        criterion = nn.BCEWithLogitsLoss()

        #evaluation loop
        model.eval()
        total_loss = 0
        count = 0

        with torch.no_grad():
            for batch in eval_loader:
                users = batch['UserID'].to(device)
                books = batch['BookID'].to(device)
                target = batch['Binary_ratings'].float().to(device)
                
                #Features added
                continous_features = torch.cat([
                    batch['book_popularity'].unsqueeze(1).to(device),
                    batch['author_popularity'].unsqueeze(1).to(device),
                    batch['publisher_popularity'].unsqueeze(1).to(device),
                    batch['user_activity'].unsqueeze(1).to(device),
                ], dim = 1)

                predictions = model(users, books, continous_features)
                
                loss = criterion(predictions, target)

                total_loss += loss.item()
                count += 1
        
        avg_loss = total_loss / count if count > 0 else float('inf')
        return avg_loss


def split_users_data(books_dataset, test_size = 0.2, val_size = 0.2):
    """
    Split data by users into train/validation/test sets
    """

    #get unique users:
    unique_users = books_dataset['UserID'].unique()

    #first split using into train+val and test
    train_val_users, test_users = train_test_split(unique_users, test_size=test_size, random_state= 42)
       
    #Then split train+val into train and val
    train_users, val_users = train_test_split(train_val_users, test_size=val_size, random_state= 42)

    #split data based on user assignments
    train_data = books_dataset[books_dataset['UserID'].isin(train_users)]
    val_data = books_dataset[books_dataset['UserID'].isin(val_users)]
    test_data = books_dataset[books_dataset['UserID'].isin(test_users)]

    return train_data, val_data, test_data

def main():
    """
    Test the model definition with a small dataset
    """
    start_time = time.time()
    try:

        #load the dataset
        books_dataset = pd.read_parquet("prepared_data/train_with_features.parquet")
        
        #convert object columns all into numeric explicitly:
        for col in books_dataset.select_dtypes(include = ['object']).columns:
            #logger.info(f"Converting {col} from Object type to Numeric type")
            books_dataset[col] = pd.to_numeric(books_dataset[col], errors='coerce')
        
        logger.info(f"Dataset is loaded and its size is: {len(books_dataset)}")
        #logger.info(f"Dataset dtypes: {books_dataset.dtypes}")

        #split data by users
        train_data, val_data, test_data = split_users_data(books_dataset)
        logger.info(f"Data split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

        #initilize and fit mappings on training data only
        data_prep = data_prepare_ml()
        data_prep.fit(train_data)

        #Transform both sets using the same mapping
        train_data = data_prep.transform(train_data)
        val_data = data_prep.transform(val_data)

        #Now get embedding sizes after mapping
        vocab_sizes = data_prep.get_vocab_sizes()

        param_distributions = {
            'n_factors': [32, 60, 100, 128],  # Added smaller and larger embedding sizes
            'learning_rate': [0.0001, 0.0005, 0.005, 0.01],  # Added smaller and larger learning rates
            'weight_decay': [0.0001, 0.005, 0.01, 0.05],  # Added more regularization options
            'batch_size': [512, 1024, 2048, 4096]  # Added smaller and larger batch sizes
        }

        #number of iterations:
        n_iter = 10 #will change depending on duration

        #randomize 
        para_list = list(ParameterSampler(param_distributions, n_iter, random_state= 42))

        #track best performance
        best_val_loss = float('inf')
        best_params = None
        best_model = None

        #try Each combination
        for i, params in enumerate(para_list):
            logger.info(f"Trial {i+1}/{n_iter}: Training with These parameters: {params}")

            #train with these parameters
            model = model_def.train_model(
                    train_data,
                    val_data,
                    vocab_sizes,
                    params['learning_rate'],
                    params['weight_decay'],
                    params['n_factors'],
                    n_epochs=10,
                    batch_size=int(params['batch_size'])
            )

            #get the final validation loss
            val_loss = model_def.calc_valid(model, val_data, int(params['batch_size']))

            #track best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = params
                best_model = model
                logger.info(f"New best model with validation loss: {best_val_loss:.4f}")

        #train the final model with these best parameters and full epochs
        logger.info(f"Training Final model with best parameters: {best_params}")
        final_model = model_def.train_model(
            train_data,
            val_data,  # Positional argument
            vocab_sizes,
            best_params['learning_rate'],  # Positional argument
            best_params['weight_decay'],  # Positional argument
            n_factors=best_params['n_factors'],  # Keyword argument
            n_epochs=10,  # Keyword argument
            batch_size=int(best_params['batch_size'])  # Keyword argument
        )


        #save the trained model
        #os.makedirs("models", exist_ok= True)
        #torch.save({
        #    'model_state_dict': model.state_dict(),
        #    'hyperparameters': best_params,
        #    '
        #}, "models/neumf_model_optimized.pt")

        
        logger.info(f"Model Training Complete")
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
        logger.info(f"Total time: {time.time() - start_time:.2f} seconds")

        return final_model
    
    except Exception as e:
        logger.error(f" Error with training model: {str(e)}")
        return None
    
if __name__ == "__main__":
    main()