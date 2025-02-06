######### building a movie recomendation system ###
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#load the movies
movies = pd.read_csv('movies.dat', sep='::', engine='python', header=None, names=['id', 'title', 'genres'])
#movies.to_csv("movies.csv")

#load the ratings
ratings = pd.read_csv("ratings.dat", sep="::", engine="python", header=None, names=["userID", "movieID", "rating", "timestamp"])
#ratings.to_csv("ratings_cleaned.csv", index=False, quoting=1)  # quoting=1 -> Quote non-numeric fields


#load the tags
tags = pd.read_csv("tags.dat", sep="::", engine = "python", header=None, names = ["userID", "movieID", "tag", "timestamp"])
#tags.to_csv("tags.csv", index=False, quoting=1)


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
genres_encoded = movie_ratings["genres"].str.get_dummies("|")   #use get_dummies to encode the genres column
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


#heatmap: correlation between user-movie features: ratings, genres, year
plt.figure(4)
numeric_columns = movie_ratings.select_dtypes(include = [np.number])
corrleation_matrix = numeric_columns.corr()
sns.heatmap(corrleation_matrix, annot = True, cmap = "coolwarm", fmt = ".2f", linewidths = 0.5)
plt.title("Heatmap of Correlation between User-Movie Features")



#numpy concepts: Basic manupulation of numpy arrays and efficient data transformation/aggregate operations

#advanced numpy concepts: broadcasting, vectorization, and array manipulation






