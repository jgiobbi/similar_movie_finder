import streamlit as st
import pandas as pd
import numpy as np
import ast

st.title('Similar Movie Finder')

df = pd.read_csv('movies_cleaned_2.csv',index_col=0)
df['genre_list'] = df['genre_list'].apply(ast.literal_eval)

def genre_similarity(genres1,genres2):
    return len(set(genres1) & set(genres2))

movie_title_input = st.text_input('Input Movie')
input_index = df['title'] == movie_title_input
input_genre_list = df[input_index]['genre_list'].iloc[0]

num_sim_genres = []
for i, row in df.iterrows():
    num_sim_genres.append(genre_similarity(row['genre_list'], input_genre_list))
    
df['num_sim_genres'] = num_sim_genres

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

knn_scaled = scaler.fit_transform(df[['year_new','avg_rating','num_sim_genres']])

input_movie = knn_scaled[input_index]

distances = []
for row in knn_scaled:
    distance = np.sqrt(np.sum((row - input_movie) ** 2))  # Euclidean distance
    distances.append(distance)

df['distance'] = distances

similar_movies_sorted = df.sort_values(by='distance').head(25)
suggested = similar_movies_sorted.iloc[1:5,1]

st.write('Suggested Similar Movies:', suggested)

