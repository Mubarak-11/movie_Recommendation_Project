# Goal of this file is to evaluate the NeuMF_model.py 

import pandas as pd, numpy as np
import logging, os, time, traceback
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import roc_auc_score
from collections import defaultdict
import torch.serialization
import torch.nn.functional as F

#import from other files
from prepare_model import book_reco_Dataset
from prepare_model import data_prepare_ml
from NeuMF_model import model_def, device


#Basic logging
logging.basicConfig(level = logging.INFO, format='%(asctime)s- %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_model(model, test_data, k_values = [5, 10, 20], batch_size = 4096, num_negatives = 99):
    """
    Evaluate model using ranking metrics on test data

    Args:
        - model: trained model
        - test_data: Test dataset
        - k_values: List of k values to calculate metrics at
        - batch_size: Batch size for evaluation
        - num_negatives: Number of negative items to sample per positive item
    
    Return:
        - Dict of metrics
    """

    model.eval()
    
    #Metrics to track
    hr_dict = defaultdict(list) #hit ratio metric
    ndcg_dict = defaultdict(list) #NDCG metric
    aucs = [] # area under curve metric

    #create test dataset
    test_dataset = book_reco_Dataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

    # Create user_interacted_dict from test_data
    user_interacted_dict = defaultdict(set)
    for _, row in test_data.iterrows():
        user_interacted_dict[row['UserID']].add(row['BookID'])


    #drive to the GPU, no need to calculate gradients again
    with torch.no_grad():
        for batch in test_loader:

            users = batch['UserID'].to(device)
            books = batch['BookID'].to(device)
            labels = batch['Binary_ratings'].float().to(device) #renamed as labels from target

            #Features added
            continous_features = torch.cat([
                batch['book_popularity'].unsqueeze(1).to(device),
                batch['author_popularity'].unsqueeze(1).to(device),
                batch['publisher_popularity'].unsqueeze(1).to(device),
                batch['user_activity'].unsqueeze(1).to(device),
            ], dim = 1)

            predictions = model(users, books, continous_features)

            # Process each user-item pair
            for i in range(len(users)):
                if labels[i] == 1:  # This is a positive example
                    user_id = users[i].item()
                    pos_item_id = books[i].item()
                    pos_score_logit = predictions[i].item()
                    pos_score = F.sigmoid(torch.tensor(pos_score_logit)).item()  # Apply sigmoid for ranking

                    # Sample negative items and get their scores
                    neg_items = sample_negative_items(user_id, pos_item_id, test_data, 
                                                    num_negatives,
                                                    num_items = model.num_items,
                                                    user_interactions = user_interacted_dict)
                    
                    # Get negative predictions in batches
                    neg_scores = []
                    for j in range(0, len(neg_items), batch_size):
                        neg_batch = neg_items[j: j+batch_size]
                        neg_users = torch.LongTensor([user_id] * len(neg_batch)).to(device)
                        neg_books = torch.LongTensor(neg_batch).to(device)

                        # Repeat continuous features for each negative item
                        neg_features = continous_features[i].repeat(len(neg_batch), 1)

                        # Get predictions for this batch
                        with torch.no_grad():
                            neg_preds = model(neg_users, neg_books, neg_features)
                        
                        # Convert to numpy and apply sigmoid in the same step
                        batch_scores = F.sigmoid(neg_preds).cpu().numpy()
                        neg_scores.extend(batch_scores)

                    # Combine pos + neg
                    all_items = [pos_item_id] + neg_items
                    all_scores = [pos_score] + neg_scores  # neg_scores already has sigmoid applied

                    # Rank items by scores
                    ranked_items = [x for _, x in sorted(zip(all_scores, all_items), reverse = True)]

                    # Hit ratio and NDCG metrics
                    for k in k_values:
                        if pos_item_id in ranked_items[:k]:
                            hr_dict[k].append(1)
                            rank = ranked_items.index(pos_item_id)
                            ndcg_dict[k].append(1 / np.log2(rank + 2))
                        else:
                            hr_dict[k].append(0)
                            ndcg_dict[k].append(0)
                    
                    # AUC metric - use the same scores that were used for ranking
                    y_true = [1] + [0] * len(neg_items)
                    y_scores = [pos_score] + neg_scores  # Use the same scores as for ranking

                    # Only calculate if we have enough samples
                    if len(neg_items) >= 5:
                        try:
                            auc = roc_auc_score(y_true, y_scores)
                            aucs.append(auc)
                        except ValueError as e:
                            logger.debug(f"AUC error: {str(e)}")

            # Aggregate results
            results = {f'HR@{k}': np.mean(hr_dict[k]) for k in k_values} 
            results.update({f'NDCG@{k}': np.mean(ndcg_dict[k]) for k in k_values})
            results['AUC'] = np.mean(aucs) if len(aucs) > 0 else float('nan')  # Handle empty list

            return results
                    

def sample_negative_items(user_id, pos_item_id, test_data, num_negatives, num_items, user_interactions):
    """
    Sample negative item IDs that the user hasn't interacted with.
    """
    # Limit number of negatives for more stable evaluation
    num_negatives = min(50, num_negatives)  # Cap at 50 negatives per positive
    
    # Get items this user has interacted with
    interacted_items = user_interactions.get(user_id, set())
    
    # All possible items minus interacted ones and the positive item
    candidates = list(set(range(num_items)) - interacted_items - {pos_item_id})
    
    # If we have fewer candidates than requested, use all available with replacement=False
    sample_size = min(num_negatives, len(candidates))
    if sample_size <= 0:
        return []  # No candidates available
        
    sample_negatives = np.random.choice(candidates, size=sample_size, replace=False)
    
    return sample_negatives.tolist()


def main():
    """
    Main function evaluate the model
    """

    start_time = time.time()
    try:

        test_data = pd.read_parquet("prepared_data/test_with_features.parquet")


        #convert any object columns to numeric
        for col in test_data.select_dtypes(include = ['object']).columns:
            test_data[col] = pd.to_numeric(test_data[col], errors = 'coerce')

        #Before loading
        torch.serialization.add_safe_globals(['numpy.core.multiarray.scalar'])

        # Load saved model to get its configuration
        model_path = "models/neumf_model_optimized.pt"
        checkpoint = torch.load(model_path, map_location=device, weights_only = False)
        
        # Extract model hyperparameters from checkpoint
        hyperparams = checkpoint.get('hyperparameters', {})
        n_factors = hyperparams.get('n_factors', 128)  # Default to 128 if not found
        
        # Get (vocab sizes) dimensions from the saved model
        n_userID = checkpoint['model_state_dict']['mf_user_factors.weight'].shape[0]
        n_BookID = checkpoint['model_state_dict']['mf_book_factors.weight'].shape[0]

       # Create model with matching architecture 
        model = model_def(n_userID=n_userID, n_BookID=n_BookID, n_factors=n_factors)
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        # Add num_items attribute for evaluation
        model.num_items = n_BookID
        
        # Evaluate model
        results = evaluate_model(model, test_data, k_values=[5, 10, 20], batch_size=4096, num_negatives=50)

        #log results
        logger.info("Evaluation Results: ")
        for metric, value in results.items():
            logger.info(f"{metric}: {value:.4f}")

        logger.info(f"Total evaluation time: {time.time() - start_time:.2f} seconds")


    except Exception as e:
        logger.error(f"Error in evaluating model {str(e)}")
        logger.error(traceback.format_exc())
        return None

if __name__== "__main__":
    main()





