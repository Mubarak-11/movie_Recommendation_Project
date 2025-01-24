######### building a movie recomendation system ###
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds




#load the movies
movies = pd.read_csv('movies.dat', sep='::', engine='python', header=None, names=['id', 'title', 'genres'])
#movies.to_csv("movies.csv")

#load the ratings
ratings = pd.read_csv("ratings.dat", sep="::", engine="python", header=None, names=["userID", "movieID", "rating", "timestamp"])
#ratings.to_csv("ratings_cleaned.csv", index=False, quoting=1)  # quoting=1 -> Quote non-numeric fields


#load the tags
tags = pd.read_csv("tags.dat", sep="::", engine = "python", header=None, names = ["userID", "movieID", "tag", "timestamp"])
#tags.to_csv("tags.csv", index=False, quoting=1)

'''
#lets check and handle missing values for all 3 csv files
print(movies.isnull().sum())
print(ratings.isnull().sum())
print(tags.isnull().sum())
print()

#remove missing rows from the tags dataframe, its missing some tags
tags.dropna(inplace=True)
tags.to_csv("tags_cleaned.csv", index=False, quoting=1)
print("Checking new tags table to see if there are any missing tags: ")
print(tags.isnull().sum())
print()

#check data types of each column are appropriate (e.g., `int` for IDs, `float` for ratings).
print(movies.dtypes)
print(ratings.dtypes)
print(tags.dtypes)
print()
'''
#movie_ratings = pd.merge(movies, ratings, on = "movieID")   #merge both movies and ratings dataframes
#movie_ratings.to_csv("Movie_ratings.csv")  # convert the dataframe to a csv file to analyze in a nice table

movie_ratings = pd.read_csv("Movie_ratings.csv")
#print(movie_ratings["movieID"].dtype)
#print(movie_ratings[movie_ratings["movieID"] == 1].shape[0])

#feature engineering 1: extract year from movie titles
movie_ratings['year'] = movie_ratings['title'].str.extract(r"\((\d{4})\)")
#print(movie_ratings.iloc[876549: 876549+5,:])
#print(movie_ratings["year"].isnull().sum()) #check for missing years

#feature engineering 2: genre encoding
genres_encoded = movie_ratings["genre"].str.get_dummies("|")   #use get_dummies to encode the genres column
movie_ratings = pd.concat([movie_ratings, genres_encoded], axis = 1)    #add the encoded columns back to the dataframe
print(movie_ratings.head(5))


#feature engineering 3: Rating statstics 
rating_stats = movie_ratings.groupby(["title"]).agg({
    "rating": ["size", "mean", "median", "std"] #size is the number of ratings
})
print(rating_stats.head(5), "\n")

#print(movie_ratings["ratings"].describe())  #get the descriptive statistics of the ratings column vs the rating_stats table

#feature engineering 4: Extract Timestamps features, converting timestamp column to datetime format
movie_ratings["timestamp"] = pd.to_datetime(movie_ratings["timestamp"], unit = "s")
print(movie_ratings[["timestamp"]].head(5), "\n")

#visualizations: use matplotlib or seaborn to create visualizations of the data
#plot a histogram of the ratings
plt.figure(1)
sns.histplot(movie_ratings["rating"], bins = 10, kde= True, color="green") 
plt.title("Distribution of Ratings")

#plot a bar chart of the number of ratings per movie
#sns.boxplot(x = "rating", data=movie_ratings, color = "blue")
#plt.title("Number of Ratings per Movie")
#plt.show()

#most rated movies
plt.figure(2)
sns.barplot(x = rating_stats["rating"]["size"].nlargest(10).values, y = rating_stats["rating"]["size"].nlargest(10).index, color="red")
plt.title("Top 10 Most Rated Movies")


#plotting user activity vs average rating with a scatter plot
plt.figure(3)
plt.figure(figsize=(10, 6))
plt.scatter(rating_stats["rating"]["size"], rating_stats["rating"]["mean"], edgecolors= "k", color="purple")
plt.title("User Activity vs Average Rating")
plt.xlabel("Number of Ratings")
plt.ylabel("Average Rating")
plt.grid(True)
#plt.show()

#heatmap: correlation between user-movie features: ratings, genres, year
plt.figure(4)
numeric_columns = movie_ratings.select_dtypes(include = [np.number])
corrleation_matrix = numeric_columns.corr()
sns.heatmap(corrleation_matrix, annot = True, cmap = "coolwarm", fmt = ".2f", linewidths = 0.5)
plt.title("Heatmap of Correlation between User-Movie Features")
#plt.show()


#numpy concepts: Basic manupulation of numpy arrays and efficient data transformation/aggregate operations

#advanced numpy concepts: broadcasting, vectorization, and array manipulation


#SVD to reduce matrices to lower dimensions, got this message hahaha: Unable to allocate 32.8 GiB for an array with shape (4407248286,)
user_item_matrix = movie_ratings.pivot(index= "userID", columns = "movieID", values= "rating").fillna(0) #create a user-item matrix, fill missing values with 0

A = user_item_matrix.values
print("Original Matrix Shape:", A.shape, "\n")
print("Original Matrix Statistics: ")
print("Min:", A.min(), "Max:", A.max(), "Mean:", A.mean())

U, sigma, Vt = svds(A, k = 100)  #apply SVD to the user-item matrix to reduce the dimensions to 50
sigma = np.diag(sigma)  #convert the sigma values to a diagonal matrix
A_k = np.dot(np.dot(U, sigma), Vt)  #reconstruct the user-item matrix with the reduced dimensions

print("Reconstructed Matrix Shape: ", A_k.shape, "\n")
print("Reconstructed Matrix Statistics:")
print("Min:", A_k.min(), "Max:", A_k.max(), "Mean:", A_k.mean(), "\n")


A_k_clip = np.clip(A_k, 0, 5) #clip the values of the reconstructed matrix to be between 0 and 5
print("Clipped Matrix Shape: ", A_k_clip.shape, "\n")
print("Clipped Matrix Statistics:")
print("Min:", A_k_clip.min(), "Max:", A_k_clip.max(), "Mean:", A_k_clip.mean())

#compute the error between the original and reconstructed matrices
error = np.sqrt(np.mean((A - A_k_clip)**2))
print("Error: ", error)

#sparse matrix: convert the user-item matrix to a sparse matrix to save memory
sparse_matrix = csr_matrix(A_k_clip) #convert the user-item matrix to a sparse matrix

#compute the cosine similarity between users
user_similarity = cosine_similarity(sparse_matrix, dense_output= False) #compute the cosine similarity between users
'''
user-user similarity matrix: rows are users, columns are users, and the values are the cosine similarity between users.
                             measures how similar users are to each other based on their ratings. Useful for recommendations based on similar users preferences
'''

#compute the cosine similarity between items
#item_similarity = cosine_similarity(sparse_matrix.T, dense_output= False) #compute the cosine similarity between items using transpose of the user-item matrix
'''
item-item similarity matrix: rows are items, columns are items, and the values are the cosine similarity between items. 
                             measures how similar items are to each other based on user ratings. Useful for recommendations based on similar items (movies) preferences
'''

#convert the user similarity matrix to a dataframe
user_similarity_df = pd.DataFrame(user_similarity, index= user_item_matrix.index, columns = user_item_matrix)
print(user_similarity_df.head(5))   #print the first 5 rows of the user similarity matrix

#convert the item similarity matrix to a dataframe 
#item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns = user_item_matrix.columns) 
#print(item_similarity_df.head(5))  #print the first 5 rows of the item similarity matrix





