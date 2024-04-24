import os
import pickle
import threading
from datetime import datetime

import joblib
import pandas as pd
from colorama import Fore, Style
from flask import Flask, request, jsonify, copy_current_request_context
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# Crear la aplicación Flask
app = Flask(__name__)

# Cargar el modelo entrenado
# model_path = "modelsft/classifier_modelV3_trained_CN_EN_ES.joblib"
# model_path = "modelsft/classifier_modelV4_trained-endpoint.joblib"
#
# model = joblib.load(model_path)
model = None  # Variable global para almacenar el modelo

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

merged_data = pd.concat([dt1, dt2, dt3, dt4], ignore_index=True)
print(merged_data.head())
# # Ordenar aleatoriamente los datos
merged_data_random = (
    merged_data[['IsDangerous', 'Contents']]
    .assign(IsDangerous=lambda x: x['IsDangerous'].astype(int, errors='ignore')).sample(frac=1, random_state=42)
)

merged_data_random.dropna()
print(merged_data_random.head())
merged_data_random.to_csv("datasets/datasetTrainMergedDataML.csv", index=False, sep=",", encoding="utf-8")


@app.route('/getModelName', methods=['GET'])
def get_model_name():
    global model  # Referencia a la variable global `model`

    # Ruta de la carpeta donde se almacenan los modelos
    models_folder = 'modelsft/'

    # Lista todos los archivos en la carpeta
    model_files = os.listdir(models_folder)

    # Filtra solo los archivos .joblib (o la extensión de los modelos)
    model_files = [file for file in model_files if file.endswith('.joblib')]

    if not model_files:
        # Si no hay archivos de modelos, retornar una respuesta indicando que no se encontraron modelos
        return jsonify({"error": "No se encontraron modelos en la carpeta 'modelsft'"}), 404

    # Ordenar los archivos por fecha de modificación, el más reciente primero
    model_files.sort(key=lambda x: os.path.getmtime(os.path.join(models_folder, x)), reverse=True)

    # El primer archivo en la lista es el más reciente
    latest_model_name = model_files[0]

    # Extraer la versión del modelo del nombre del archivo
    # Supongamos que los nombres de archivo siguen el formato: model_vN.joblib
    # donde N es la versión del modelo.
    model_version = latest_model_name.split('_')[-2].replace('.joblib', '')

    # Cargar el modelo desde el archivo más reciente
    model_path = os.path.join(models_folder, latest_model_name)
    model = joblib.load(model_path)

    # Retornar el nombre del archivo del último modelo y su versión
    return jsonify({
        "latest_model_name": latest_model_name,
        "model_version": model_version
    }), 200


# @app.route('/train', methods=['POST'])
# def train_and_evaluate_model():
#     # Obtener los datos de la solicitud HTTP
#     data_file = request.files.get('dataset')
#     model_name = request.form.get('model_name')
#
#     # Verifica si se recibieron los datos correctamente
#     if not data_file or not model_name:
#         return jsonify({"error": "Falta el archivo del dataset o el nombre del nuevo modelo"}), 400
#
#     # Guardar el archivo en el disco temporalmente (opcional)
#     dataset_path = f"datasets/temp/{data_file.filename}"
#     data_file.save(dataset_path)
#
#     # Leer el dataset directamente desde el archivo en memoria
#     dt = pd.read_csv(dataset_path)
#     data = pd.concat([dt, merged_data_random])
#     print(f"{Fore.GREEN}[MERGE]{Style.RESET_ALL} Uniendo dataset original con el dataset cargado...")
#     # Eliminar filas con valores NaN
#     print(f"{Fore.YELLOW}[CLEAN]{Style.RESET_ALL} Eliminando filas con valores NaN")
#     data.dropna(subset=['Contents', 'IsDangerous'], inplace=True)
#
#     # Dividir los datos en conjuntos de entrenamiento y prueba
#     print(f"{Fore.BLUE}[SPLIT]{Style.RESET_ALL} Dividiendo los datos en conjuntos de entrenamiento y prueba")
#     X_train, X_test, y_train, y_test = train_test_split(data['Contents'], data['IsDangerous'], test_size=0.2,
#                                                         random_state=42)
#
#     # Crear un pipeline con un vectorizador TF-IDF y un clasificador SVM
#     print(
#         f"{Fore.MAGENTA}[PIPELINE]{Style.RESET_ALL} Creando un pipeline con un vectorizador TF-IDF y un clasificador SVM")
#     pipeline = Pipeline([
#         ('tfidf', TfidfVectorizer()),
#         ('svm', SVC())
#     ])
#
#     # Entrenar el pipeline
#     print(f"{Fore.RED}[TRAIN]{Style.RESET_ALL} Entrenando el modelo...")
#     pipeline.fit(X_train, y_train)
#
#     # Realizar predicciones en el conjunto de prueba
#     print(f"{Fore.CYAN}[PREDICT]{Style.RESET_ALL} Realizando predicciones en el conjunto de prueba")
#     predictions = pipeline.predict(X_test)
#
#     # Generar el informe de estadísticas del modelo
#     print(f"{Fore.GREEN}[REPORT]{Style.RESET_ALL} Generando informe de estadísticas del modelo")
#     report = classification_report(y_test, predictions)
#
#     # Guardar el modelo entrenado
#     model_path = f"modelsft/{model_name}.joblib"
#     print(f"{Fore.YELLOW}[SAVE]{Style.RESET_ALL} Guardando el modelo entrenado en {model_path}")
#     joblib.dump(pipeline, model_path)
#
#     # Mostrar el informe completo
#     print(f"{Fore.MAGENTA}--------------- Model Statistics Report {model_path} ---------------{Style.RESET_ALL}")
#     print(report)
#
#     # Devuelve una respuesta JSON con la URL del modelo guardado (si es necesario)
#     return jsonify({"message": "Entrenamiento completado con éxito", "model_path": model_path}), 200

# @app.route('/train', methods=['POST'])
# def train_and_evaluate_model():
#     global model  # Referencia a la variable global `model`
#
#     # Obtener los datos de la solicitud HTTP
#     data_file = request.files.get('dataset')
#
#     # Verifica si se recibió el archivo correctamente
#     if not data_file:
#         return jsonify({"error": "Falta el archivo del dataset"}), 400
#
#     # Guardar el archivo en el disco temporalmente (opcional)
#     dataset_path = f"datasets/temp/{data_file.filename}"
#     data_file.save(dataset_path)
#
#     # Leer el dataset directamente desde el archivo en memoria
#     dt = pd.read_csv(dataset_path)
#     data = pd.concat([dt, merged_data_random])
#     print(f"{Fore.GREEN}[MERGE]{Style.RESET_ALL} Uniendo dataset original con el dataset cargado...")
#     # Eliminar filas con valores NaN
#     print(f"{Fore.YELLOW}[CLEAN]{Style.RESET_ALL} Eliminando filas con valores NaN")
#     data.dropna(subset=['Contents', 'IsDangerous'], inplace=True)
#
#     # Dividir los datos en conjuntos de entrenamiento y prueba
#     print(f"{Fore.BLUE}[SPLIT]{Style.RESET_ALL} Dividiendo los datos en conjuntos de entrenamiento y prueba")
#     X_train, X_test, y_train, y_test = train_test_split(data['Contents'], data['IsDangerous'], test_size=0.2,
#                                                         random_state=42)
#
#     # Crear un pipeline con un vectorizador TF-IDF y un clasificador SVM
#     print(
#         f"{Fore.MAGENTA}[PIPELINE]{Style.RESET_ALL} Creando un pipeline con un vectorizador TF-IDF y un clasificador SVM")
#     pipeline = Pipeline([
#         ('tfidf', TfidfVectorizer()),
#         ('svm', SVC())
#     ])
#
#     # Entrenar el pipeline
#     print(f"{Fore.RED}[TRAIN]{Style.RESET_ALL} Entrenando el modelo...")
#     pipeline.fit(X_train, y_train)
#
#     # Realizar predicciones en el conjunto de prueba
#     print(f"{Fore.CYAN}[PREDICT]{Style.RESET_ALL} Realizando predicciones en el conjunto de prueba")
#     predictions = pipeline.predict(X_test)
#
#     # Generar el informe de estadísticas del modelo
#     print(f"{Fore.GREEN}[REPORT]{Style.RESET_ALL} Generando informe de estadísticas del modelo")
#     report = classification_report(y_test, predictions)
#
#     # Determinar la versión del último modelo y calcular la nueva versión
#     models_folder = 'modelsft/'
#     model_files = os.listdir(models_folder)
#     model_files = [file for file in model_files if file.endswith('.joblib')]
#
#     if model_files:
#         # Ordenar los archivos por fecha de modificación, el más reciente primero
#         model_files.sort(key=lambda x: os.path.getmtime(os.path.join(models_folder, x)), reverse=True)
#
#         # Extraer la versión del último modelo
#         latest_model_name = model_files[0]
#         latest_version_str = latest_model_name.split('_')[1]  # Suponiendo el formato 'classifier_modelV#'
#         latest_version = int(latest_version_str.replace('V', ''))
#
#         # Calcular la nueva versión sumando 1
#         new_version = latest_version + 1
#     else:
#         # Si no hay modelos previos, empezamos con la versión 1
#         new_version = 1
#
#     # Generar el nombre del nuevo modelo con la nueva versión y la fecha y hora actual
#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#     new_model_name = f"classifier_modelV{new_version}_{timestamp}.joblib"
#     model_path = os.path.join(models_folder, new_model_name)
#
#     # Guardar el modelo entrenado
#     print(f"{Fore.YELLOW}[SAVE]{Style.RESET_ALL} Guardando el modelo entrenado en {model_path}")
#     joblib.dump(pipeline, model_path)
#
#     # Mostrar el informe completo
#     print(f"{Fore.MAGENTA}--------------- Model Statistics Report {model_path} ---------------{Style.RESET_ALL}")
#     print(report)
#
#     # Devuelve una respuesta JSON con la confirmación del éxito del entrenamiento y la ruta del modelo guardado
#     return jsonify({"message": "Entrenamiento completado con éxito", "model_path": model_path})

@app.route('/train', methods=['POST'])
def train_and_evaluate_model():
    def process_training_request():
        try:
            # Obtener los datos de la solicitud HTTP
            data_file = request.files.get('dataset')

            # Verifica si se recibió el archivo correctamente
            if not data_file:
                print("Error: Falta el archivo del dataset")
                return {"error": "Falta el archivo del dataset"}

            # Guardar el archivo en el disco temporalmente
            dataset_path = f"datasets/temp/{data_file.filename}"
            data_file.save(dataset_path)

            # Leer el dataset desde el archivo
            dt = pd.read_csv(dataset_path)

            # Puedes añadir el código de fusión aquí si tienes un dataset adicional para combinar
            # data = pd.concat([dt, merged_data_random])

            # Eliminar filas con valores NaN
            dt.dropna(subset=['Contents', 'IsDangerous'], inplace=True)

            # Dividir los datos en conjuntos de entrenamiento y prueba
            X_train, X_test, y_train, y_test = train_test_split(dt['Contents'], dt['IsDangerous'], test_size=0.2,
                                                                random_state=42)

            # Crear un pipeline con un vectorizador TF-IDF y un clasificador SVM
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer()),
                ('svm', SVC())
            ])

            # Entrenar el pipeline
            pipeline.fit(X_train, y_train)

            # Realizar predicciones en el conjunto de prueba
            predictions = pipeline.predict(X_test)

            # Generar el informe de estadísticas del modelo
            report = classification_report(y_test, predictions)

            # Manejar la versión del modelo
            models_folder = 'modelsft/'
            model_files = [file for file in os.listdir(models_folder) if file.endswith('.joblib')]

            if model_files:
                # Ordenar los archivos por fecha de modificación, el más reciente primero
                model_files.sort(key=lambda x: os.path.getmtime(os.path.join(models_folder, x)), reverse=True)

                # Extraer la versión del último modelo
                latest_model_name = model_files[0]
                latest_version_str = latest_model_name.split('_')[1]  # Formato 'classifier_modelV#'
                latest_version = int(latest_version_str.replace('V', ''))

                # Calcular la nueva versión sumando 1
                new_version = latest_version + 1
            else:
                # Si no hay modelos previos, empezamos con la versión 1
                new_version = 1

            # Generar el nombre del nuevo modelo con la nueva versión y la fecha y hora actual
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            new_model_name = f"classifier_modelV{new_version}_{timestamp}.joblib"
            model_path = os.path.join(models_folder, new_model_name)

            # Guardar el modelo entrenado
            joblib.dump(pipeline, model_path)

            # Mostrar el informe completo
            print(f"--------------- Model Statistics Report {model_path} ---------------")
            print(report)
            # Al finalizar el entrenamiento, devolver un diccionario con el mensaje de éxito y la ruta del modelo
            return {"message": "Entrenamiento completado con éxito", "model_path": model_path}

        except Exception as e:
            print(f"Error durante el proceso de entrenamiento: {str(e)}")
            # Manejar el error devolviendo un diccionario con el error
            return {"error": f"Error durante el entrenamiento: {str(e)}"}

    # Crear un hilo para manejar la solicitud de entrenamiento
    training_thread = threading.Thread(target=process_training_request)
    training_thread.start()

    # Obtener el resultado de process_training_request
    result = process_training_request()

    # Responder con un código de estado HTTP 202 (Accepted) indicando que el entrenamiento está en proceso
    # o con un código de estado HTTP 200 (OK) si se completó el entrenamiento
    if "error" in result:
        return jsonify(result), 500
    else:
        return jsonify(result), 200


# Función para calcular la similitud coseno entre dos textos
def calculate_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
    return similarity


@app.route('/mlclassifier', methods=['POST'])
def ml_classifier():
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
        print(
            f"{content}: {Fore.RED + '[Peligroso]' if prediction == 1 else Fore.GREEN + '[No Peligroso]'}{Style.RESET_ALL}")

    save_cache_to_disk()

    return jsonify(predictions)


if __name__ == '__main__':
    app.run(debug=True)
