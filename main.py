from fastapi import FastAPI
import pandas as pd
import pyarrow.parquet as pq

app = FastAPI()

# Cargar las tablas desde los archivos parquet
max_playtime_per_genre = pq.read_table("max_playtime_per_genre.parquet").to_pandas()
user_total_playtime_general = pq.read_table("user_total_playtime_general.parquet").to_pandas()
top_3_games_per_year = pq.read_table("top_3_games_per_year.parquet").to_pandas()
bottom_3_games_per_year = pq.read_table("bottom_3_games_per_year.parquet").to_pandas()
sentiment_counts_sorted = pq.read_table("sentiment_counts_sorted.parquet").to_pandas()
cosine_sim_df = pq.read_table("cosine_sim_df.parquet").to_pandas()

@app.get("/PlayTimeGenre/{genre}")
def PlayTimeGenre(genre: str):
    # Verificar si el género se encuentra en la columna genres del DataFrame max_playtime_per_genre
    if genre not in max_playtime_per_genre['genres'].unique():
        return f"El género '{genre}' no se encuentra en la base de datos."
    
    # Filtrar los juegos que correspondan al género proporcionado
    genre_data = max_playtime_per_genre.dropna(subset=['genres'])
    genre_data = genre_data[genre_data['genres'].apply(lambda x: genre in x)]

    # Encontrar el año con la mayor cantidad de horas jugadas
    max_year = genre_data.loc[genre_data['playtime_forever'].idxmax()]['year']

    return f"El año con más horas jugadas para el género {genre} es: {max_year}"

@app.get("/UserForGenre/{genre}")
def UserForGenre(genre: str):
    
    # Verificar si el género se encuentra en la columna genres del DataFrame
    if genre not in user_total_playtime_general['genres'].unique():
        return f"El género '{genre}' no se encuentra en la base de datos."
    
    # Filtrar los juegos que correspondan al género proporcionado
    genre_data = user_total_playtime_general[user_total_playtime_general['genres'].apply(lambda x: genre in x)]
        
    # Agrupar por usuario y calcular la suma de las horas jugadas para cada usuario
    user_total_playtime = genre_data.groupby('user_id')['playtime_forever'].sum()
    user_total_playtime = user_total_playtime.reset_index()
    
    # Obtener el usuario con más horas jugadas
    sorted_users = user_total_playtime.sort_values(by='playtime_forever', ascending=False)
    sorted_users = sorted_users.reset_index()
    sorted_users.drop('index', axis=1, inplace=True)
    
    # Obtener el usuario con más horas jugadas
    max_user = sorted_users['user_id'].iloc[0]
    
    # Filtrar los datos solo para el usuario con la máxima cantidad de horas jugadas
    max_user_data = genre_data[genre_data['user_id'] == max_user]

    # Agrupar por año y calcular la suma de las horas jugadas para ese usuario
    year_playtime = max_user_data.groupby('year')['playtime_forever'].sum().reset_index()
    
    # Convertir a lista de listas (año, horas acumuladas)
    year_playtime_list = year_playtime.values.tolist()

    return max_user, year_playtime_list

@app.get("/UsersRecommend/{year}")
def UsersRecommend(year: str):
    # Verificar si el año se encuentra en la columna year del DataFrame top_3_games_per_year
    if str(year) not in top_3_games_per_year['year'].unique():
        return f"El año '{year}' no se encuentra en la base de datos."

    # Filtrar los juegos del año especificado
    top_3 = top_3_games_per_year[top_3_games_per_year['year'] == str(year)]
    
    # Ordenar los juegos basados en la suma de sentiment_analysis en orden descendente
    recommended_games = top_3.sort_values(by='sentiment_analysis', ascending=False).reset_index().head(3)
    
    # Obtener los nombres de los juegos recomendados
    recommended_games_names = recommended_games['app_name'].tolist()
    
    # Formatear el resultado
    result = [{"Puesto {}".format(i + 1): game} for i, game in enumerate(recommended_games_names)]
    
    return result


@app.get("/UsersNotRecommend/{year}")
def UsersNotRecommend(year: str):
    
    # Verificar si el año se encuentra en la columna year del DataFrame bottom_3_games_per_year
    if str(year) not in bottom_3_games_per_year['year'].unique():
        return f"El año '{year}' no se encuentra en la base de datos."
    # Filtrar los juegos del año especificado
    
    bottom_3 = bottom_3_games_per_year[bottom_3_games_per_year['year'] == str(year)]
    
    # Ordenar los juegos basados en la suma de count en orden descendente
    not_recommended_games = bottom_3.sort_values(by='count', ascending=False).reset_index().head(3)

    # Obtener los nombres de los juegos recomendados
    not_recommended_games = not_recommended_games['app_name'].tolist()
    
    # Formatear el resultado
    result = [{"Puesto {}".format(i + 1): game} for i, game in enumerate(not_recommended_games)]
    
    return result

@app.get("/sentiment_analysis/{year}")
def sentiment_analysis(year: str):

    # Verificar si el año se encuentra en la columna year del DataFrame sentiment_counts_sorted
    if year not in sentiment_counts_sorted['year'].unique():
        return f"El año '{year}' no se encuentra en la base de datos."

    # Filtrar las reseñas del año especificado
    reviews = sentiment_counts_sorted[sentiment_counts_sorted['year'] == (year)]
    
    # Crear un diccionario con valores predeterminados
    sentiment_dict = {'Negative': 0, 'Neutral': 0, 'Positive': 0}

    # Actualizar los valores del diccionario con los valores del DataFrame
    for _, row in reviews.iterrows():
        sentiment = row['sentiment_analysis']
        if sentiment == 0:
            sentiment_dict['Negative'] = row['count']
        elif sentiment == 1:
            sentiment_dict['Neutral'] = row['count']
        elif sentiment == 2:
            sentiment_dict['Positive'] = row['count']

    return sentiment_dict

@app.get("/recomendacion_juego/{juego}")
def recomendacion_juego(juego: str):
    # Verificar si el juego está en el DataFrame
    if juego not in cosine_sim_df.index:
        return(f"El juego '{juego}' no se encuentra en la base de datos.")
    
    # Obtener la fila de similitud del juego dado
    sim_row = cosine_sim_df.loc[juego]
    
    # Ordenar las similitudes en orden descendente y obtener los índices de los juegos más similares
    most_similar_indices = sim_row.drop(juego).sort_values(ascending=False).head(5).index
    
    # Crear una lista con los juegos más similares
    juegos_similares = [juego_similar for juego_similar in most_similar_indices]
    
    return juegos_similares