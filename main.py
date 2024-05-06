from fastapi import FastAPI
import pandas as pd
import pyarrow.parquet as pq

app = FastAPI()

# Cargar las tablas desde los archivos parquet
steam_games = pq.read_table("steam_games.parquet").to_pandas()
users_items = pq.read_table("users_items.parquet").to_pandas()
user_reviews = pq.read_table("user_reviews.parquet").to_pandas()
max_playtime_per_genre = pq.read_table("max_playtime_per_genre.parquet").to_pandas()
user_total_playtime_general = pq.read_table("user_total_playtime_general.parquet").to_pandas()

@app.get("/PlayTimeGenre/{genre}")
def PlayTimeGenre(genre: str):
    
    # Filtrar los juegos que correspondan al género proporcionado
    genre_data = max_playtime_per_genre.dropna(subset=['genres'])
    genre_data = genre_data[genre_data['genres'].apply(lambda x: genre in x)]

    # Encontrar el año con la mayor cantidad de horas jugadas
    max_year = genre_data.loc[genre_data['playtime_forever'].idxmax()]['year']

    return f"El año con más horas jugadas para el género {genre} es: {max_year}"

@app.get("/UserForGenre/{genre}")
def UserForGenre(genre: str):

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
    # Realizar un left merge entre user_reviews y steam_games
    merged_data = pd.merge(user_reviews[['item_id', 'recommend', 'sentiment_analysis']],
                           steam_games[['id', 'app_name', 'release_date']],
                           left_on='item_id',
                           right_on='id',
                           how='left')
    # Eliminando columnas innecesarias y datos nulos
    merged_data.drop(['id', 'item_id'], axis=1, inplace=True)
    merged_data = merged_data.dropna(subset=['release_date'])
    
    # Filtrar los juegos del año especificado
    # Extraer el año de la columna "release_date"
    merged_data['year'] = merged_data['release_date'].str.extract(r'(\d{4})|(\w{3}\s(\d{4}))')[0].fillna(merged_data['release_date'].str.extract(r'(\d{4})|(\w{3}\s(\d{4}))')[2])
    merged_data = merged_data[merged_data['year'] == str(year)]

    # Filtrar los juegos recomendados con sentiment_analysis de 1 o 2
    recommended_games = merged_data[(merged_data['recommend'] == True) & 
                                    (merged_data['sentiment_analysis'].isin([1, 2]))]
    
    # Agrupar por app_name y sumar los valores de sentiment_analysis
    grouped = recommended_games.groupby('app_name')['sentiment_analysis'].sum().reset_index()
    
    # Ordenar los juegos basados en la suma de sentiment_analysis en orden descendente
    recommended_games = grouped.sort_values(by='sentiment_analysis', ascending=False).reset_index().head(3)
    
    # Obtener los nombres de los juegos recomendados
    recommended_games_names = recommended_games['app_name'].tolist()
    
    # Formatear el resultado
    result = [{"Puesto {}".format(i + 1): game} for i, game in enumerate(recommended_games_names)]
    
    return result


@app.get("/UsersNotRecommend/{year}")
def UsersNotRecommend(year: str):
    # Realizar un left merge entre user_reviews y steam_games
    merged_data = pd.merge(user_reviews[['item_id', 'recommend', 'sentiment_analysis']],
                           steam_games[['id', 'app_name', 'release_date']],
                           left_on='item_id',
                           right_on='id',
                           how='left')
    # Eliminando columnas innecesarias y datos nulos
    merged_data.drop(['id', 'item_id'], axis=1, inplace=True)
    merged_data = merged_data.dropna(subset=['release_date'])
    
    # Filtrar los juegos del año especificado
    # Extraer el año de la columna "release_date"
    merged_data['year'] = merged_data['release_date'].str.extract(r'(\d{4})|(\w{3}\s(\d{4}))')[0].fillna(merged_data['release_date'].str.extract(r'(\d{4})|(\w{3}\s(\d{4}))')[2])
    merged_data = merged_data[merged_data['year'] == str(year)]

    # Filtrar los juegos recomendados con sentiment_analysis de 0
    not_recommended_games = merged_data[(merged_data['recommend'] == False) & 
                                    (merged_data['sentiment_analysis'].isin([0]))]
    
    # Agrupar por app_name y sumar los valores de sentiment_analysis
    grouped = not_recommended_games.groupby('app_name').agg(
        sentiment_analysis=('sentiment_analysis', 'sum'),
        count=('sentiment_analysis', 'count')
    ).reset_index()
    
    # Ordenar los juegos basados en la suma de sentiment_analysis en orden descendente
    not_recommended_games = grouped.sort_values(by='count', ascending=False).reset_index().head(3)

    # Obtener los nombres de los juegos recomendados
    not_recommended_games = not_recommended_games['app_name'].tolist()
    
    # Formatear el resultado
    result = [{"Puesto {}".format(i + 1): game} for i, game in enumerate(not_recommended_games)]
    
    return result

@app.get("/sentiment_analysis/{year}")
def sentiment_analysis(year: str):
    # Agregar la columna "release_date" al dataframe "user_reviews" mediante un merge
    reviews_with_release = pd.merge(user_reviews, steam_games[['id', 'release_date']], 
                                    left_on='item_id', right_on='id', how='left')
    
    # Filtrar las reseñas del año especificado
    reviews_with_release['year'] = reviews_with_release['release_date'].str.extract(r'(\d{4})|(\w{3}\s(\d{4}))')[0].fillna(reviews_with_release['release_date'].str.extract(r'(\d{4})|(\w{3}\s(\d{4}))')[2])
    reviews_with_release = reviews_with_release[reviews_with_release['year'] == str(year)]
    
    # Contar las reseñas por categoría de sentimiento
    sentiment_counts = reviews_with_release['sentiment_analysis'].value_counts()
    
    # Convertir los valores de numpy.int32 a int
    sentiment_counts = sentiment_counts.astype(int)
    
    # Crear un diccionario con los conteos de cada categoría de sentimiento
    result = {
        'Negative': int(sentiment_counts.get(0, 0)),
        'Neutral': int(sentiment_counts.get(1, 0)),
        'Positive': int(sentiment_counts.get(2, 0))
    }

    return result
