import pickle

import joblib
import pandas as pd
from colorama import Fore, Style
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# Crear la aplicación Flask
app = Flask(__name__)

# Cargar el modelo entrenado
model_path = "modelsft/classifier_modelV3_trained_CN_EN_ES.joblib"
model = joblib.load(model_path)

# Cargar el diccionario cache desde el disco si existe, o crear uno nuevo si no existe
try:
    with open('cache.pkl', 'rb') as f:
        cache = pickle.load(f)
except FileNotFoundError:
    cache = {}


def save_cache_to_disk():
    with open('cache.pkl', 'wb') as f:
        pickle.dump(cache, f)
    export_cache_to_csv()


def export_cache_to_csv():
    # Convierte el diccionario de caché en un DataFrame de pandas
    cache_df = pd.DataFrame(list(cache.items()), columns=['content', 'prediction'])
    # Guarda el DataFrame como un archivo CSV en la carpeta 'datasets'
    cache_df.to_csv("datasets/datasetCachePredictions.csv", index=False, sep=",")
    print(f"{Fore.GREEN}[SAVE]{Style.RESET_ALL} Caché exportado a CSV en datasets/datasetCachePredictions.csv")


# # Leer el dataset
dt1 = pd.read_csv("datasets/datasetTrainMergedData.csv", sep=",")
dt2 = pd.read_csv("datasets/datasetV2.csv", encoding="latin-1", sep=",")
dt3 = pd.read_csv("datasets/datasetV2_EN.csv", sep=",")
dt4 = pd.read_csv("datasets/datasetV2_CN.csv", sep=",", encoding="utf-8")
dt4.rename(columns={'很危险': 'IsDangerous', '内容': 'Contents'}, inplace=True)

print(dt1.head())
print(dt2.head())
print(dt3.head())
print(dt4.head())

merged_data = pd.concat([dt1, dt2, dt3, dt4], ignore_index=True)
print(merged_data.head())
# # Ordenar aleatoriamente los datos
merged_data_random = (
    merged_data[['IsDangerous', 'Contents']]
    .assign(IsDangerous=lambda x: x['IsDangerous'].astype(int, errors='ignore')).sample(frac=1,
                                                                                        random_state=42)
)

# merged_data_random = merged_data_random[['IsDangerous', 'Contents']]
#
# # Mostrar información sobre el conjunto de datos fusionado y ordenado aleatoriamente
# print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} dimensiones del dataset  {merged_data_random.shape}")
# print("\nPrimeras filas del conjunto de datos fusionado y ordenado aleatoriamente:")
print(merged_data_random.head())
merged_data_random.to_csv("datasets/datasetTrainMergedDataML.csv", index=False, sep=",", encoding="utf-8")


@app.route('/train')
def train_and_evaluate_model(dataset_path, model_path):
    # Log de inicio del entrenamiento
    print(
        f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Comenzando el entrenamiento del modelo usando el dataset en {dataset_path}")

    # Cargar el dataset
    print(f"{Fore.GREEN}[LOAD]{Style.RESET_ALL} Cargando el dataset desde {dataset_path}")
    data = pd.read_csv(dataset_path)

    # Eliminar filas con valores NaN
    print(f"{Fore.YELLOW}[CLEAN]{Style.RESET_ALL} Eliminando filas con valores NaN")
    data.dropna(subset=['Contents', 'IsDangerous'], inplace=True)

    # Dividir los datos en conjuntos de entrenamiento y prueba
    print(f"{Fore.BLUE}[SPLIT]{Style.RESET_ALL} Dividiendo los datos en conjuntos de entrenamiento y prueba")
    X_train, X_test, y_train, y_test = train_test_split(data['Contents'], data['IsDangerous'], test_size=0.3,
                                                        random_state=42)

    # Crear un pipeline con un vectorizador TF-IDF y un clasificador SVM
    print(
        f"{Fore.MAGENTA}[PIPELINE]{Style.RESET_ALL} Creando un pipeline con un vectorizador TF-IDF y un clasificador SVM")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('svm', SVC())
    ])

    # Entrenar el pipeline
    print(f"{Fore.RED}[TRAIN]{Style.RESET_ALL} Entrenando el modelo...")
    pipeline.fit(X_train, y_train)

    # Realizar predicciones en el conjunto de prueba
    print(f"{Fore.CYAN}[PREDICT]{Style.RESET_ALL} Realizando predicciones en el conjunto de prueba")
    predictions = pipeline.predict(X_test)

    # Generar el informe de estadísticas del modelo
    print(f"{Fore.GREEN}[REPORT]{Style.RESET_ALL} Generando informe de estadísticas del modelo")
    report = classification_report(y_test, predictions)

    # Guardar el modelo
    print(f"{Fore.YELLOW}[SAVE]{Style.RESET_ALL} Guardando el modelo entrenado en {model_path}")
    joblib.dump(pipeline, model_path)

    # Mostrar el informe completo
    print(f"{Fore.MAGENTA}--------------- Model Statistics Report {dataset_path} ---------------{Style.RESET_ALL}")
    print(report)

    # Guardar la caché
    print(f"{Fore.BLUE}[CACHE]{Style.RESET_ALL} Guardando la caché en disco")
    save_cache_to_disk()

    # Notificar que el entrenamiento ha finalizado
    print(f"{Fore.RED}[INFO]{Style.RESET_ALL} Entrenamiento ha finalizado")


# Entrenar y evaluar el modelo con los datasets proporcionados
# train_and_evaluate_model("datasets/datasetTrainMergedData.csv", "modelsft/classifier_modelV3_trained_CN_EN_ES.joblib")


# Función para calcular la similitud coseno entre dos textos
def calculate_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
    return similarity


def update_cache():
    cache["HIDRUROS"] = 1
    cache["CUCHILLOS"] = 1
    cache["Cuchillos"] = 1
    cache["GUNS"] = 1
    cache["AMMO"] = 1
    save_cache_to_disk()


@app.route('/mlclassifier', methods=['POST'])
def ml_classifier():
    update_cache()
    content_list = request.json.get('contents', [])
    predictions = []

    for content in content_list:
        similar_cached_predictions = [cache.get(word.lower(), -1) for word in content.split()]
        similar_cached_predictions = [int(prediction) for prediction in similar_cached_predictions if prediction != -1]

        if similar_cached_predictions:
            prediction = max(set(similar_cached_predictions), key=similar_cached_predictions.count)
        else:
            prediction = model.predict([content])[0]
            cache[content.lower()] = prediction

        prediction = int(prediction)

        predictions.append({'content': content, 'prediction': prediction})
        print(f"{content}: {'Peligroso' if prediction == 1 else 'No Peligroso'}")

    save_cache_to_disk()

    return jsonify(predictions)


if __name__ == '__main__':
    app.run(debug=True)
