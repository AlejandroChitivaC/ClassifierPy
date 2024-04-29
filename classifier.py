import os
import pickle
import threading
from datetime import datetime
from queue import Queue

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

model = None

try:
    with open('cache.pkl', 'rb') as f:
        cache = pickle.load(f)
except FileNotFoundError:
    cache = {}


def save_cache_to_disk():
    with open('cache.pkl', 'wb') as f:
        pickle.dump(cache, f)


# # Leer el dataset
dt1 = pd.read_csv("datasets/datasetTrainMergedData.csv", sep=",")
dt2 = pd.read_csv("datasets/datasetV2.csv", encoding="latin-1", sep=",")
dt3 = pd.read_csv("datasets/datasetV2_EN.csv", sep=",")
dt4 = pd.read_csv("datasets/datasetV2_CN.csv", sep=",", encoding="utf-8")
dt4.rename(columns={'很危险': 'IsDangerous', '内容': 'Contents'}, inplace=True)

merged_data = pd.concat([dt1, dt2, dt3, dt4], ignore_index=True)
# # Ordenar aleatoriamente los datos
merged_data_random = (
    merged_data[['IsDangerous', 'Contents']]
    .assign(IsDangerous=lambda x: x['IsDangerous'].astype(int, errors='ignore')).sample(frac=1, random_state=42)
)

merged_data_random.dropna()
merged_data_random.to_csv("datasets/datasetTrainMergedDataML.csv", index=False, sep=",", encoding="utf-8")


@app.route('/getModelName', methods=['GET'])
def get_model_name():
    global model

    models_folder = 'modelsft/'
    model_files = os.listdir(models_folder)
    model_files = [file for file in model_files if file.endswith('.joblib')]

    if not model_files:
        return jsonify({"error": "No se encontraron modelos en la carpeta 'modelsft'"}), 404

    model_files.sort(key=lambda x: os.path.getmtime(os.path.join(models_folder, x)), reverse=True)
    latest_model_name = model_files[0]
    model_version = latest_model_name.split('_')[-2].replace('.joblib', '')

    model_path = os.path.join(models_folder, latest_model_name)
    model = joblib.load(model_path)

    return jsonify({
        "latest_model_name": latest_model_name,
        "model_version": model_version
    }), 200


@app.route('/train', methods=['POST'])
def train_and_evaluate_model():
    clear_cache()
    data_file = request.files.get('dataset')
    if not data_file:
        return jsonify({"error": "Falta el archivo del dataset"}), 400

    result_queue = Queue()
    training_thread = threading.Thread(target=process_training_request, args=(data_file, result_queue))
    training_thread.start()

    result = result_queue.get()

    # Verificar si hubo algún error durante el entrenamiento
    if "error" in result:
        return jsonify(result), 500
    else:
        return jsonify(result), 200


def process_training_request(data_file, result_queue):
    try:
        global model

        dataset_path = f"datasets/temp/{data_file.filename}"
        data_file.save(dataset_path)

        new_data = pd.read_csv(dataset_path)

        print(f"{Fore.GREEN}[MERGE]{Style.RESET_ALL} Combinando dataset original con el nuevo dataset cargado...")
        combined_data = pd.concat([merged_data_random, new_data], ignore_index=True)
        print(f"{Fore.YELLOW}[CLEAN]{Style.RESET_ALL} Eliminando filas con valores NaN")
        combined_data.dropna(subset=['Contents', 'IsDangerous'], inplace=True)

        print(f"{Fore.BLUE}[SPLIT]{Style.RESET_ALL} Dividiendo los datos en conjuntos de entrenamiento y prueba")
        X_train, X_test, y_train, y_test = train_test_split(combined_data['Contents'], combined_data['IsDangerous'],
                                                            test_size=0.2, random_state=42)

        print(f"{Fore.BLUE}[INFO]{Style.RESET_ALL} Obteniendo versión del último modelo entrenado")
        models_folder = 'modelsft/'
        model_files = [file for file in os.listdir(models_folder) if file.endswith('.joblib')]

        if model_files:
            model_files.sort(key=lambda x: os.path.getmtime(os.path.join(models_folder, x)), reverse=True)

            latest_model_name = model_files[0]
            print(f"{Fore.LIGHTGREEN_EX}[LOAD]{Style.RESET_ALL} Último modelo encontrado: {latest_model_name}")

            latest_model_path = os.path.join(models_folder, latest_model_name)

            print(f"{Fore.BLUE}[INFO]{Style.RESET_ALL} Último modelo cargado: {latest_model_name}")
            model = joblib.load(latest_model_path)

        else:
            model = Pipeline([
                ('tfidf', TfidfVectorizer()),
                ('svm', SVC())
            ])
            print(
                f"{Fore.MAGENTA}[PIPELINE]{Style.RESET_ALL} No se encontró un modelo previo, creando un nuevo pipeline")

        # Reentrenar el modelo
        print(f"{Fore.LIGHTRED_EX}[TRAIN]{Style.RESET_ALL} Reentrenando el modelo con el nuevo conjunto de datos...")
        model.fit(X_train, y_train)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        new_model_path = f"modelsft/retrained_{latest_model_name.replace('.joblib', '')}_{timestamp}_trainedAzure.joblib"

        # Guardar el modelo reentrenado
        print(f"{Fore.YELLOW}[SAVE]{Style.RESET_ALL} Guardando el modelo reentrenado en {new_model_path}")
        joblib.dump(model, new_model_path)

        # Realizar predicciones en el conjunto de prueba
        print(f"{Fore.CYAN}[PREDICT]{Style.RESET_ALL} Realizando predicciones en el conjunto de prueba")
        predictions = model.predict(X_test)

        # Generar el informe de estadísticas del modelo
        print(f"{Fore.GREEN}[REPORT]{Style.RESET_ALL} Generando informe de estadísticas del modelo")
        report = classification_report(y_test, predictions)
        print(report)

        print(
            f"{Fore.MAGENTA}--------------- Model Statistics Report {new_model_path} ---------------{Style.RESET_ALL}")
        print(report)

        # Guardar el resultado del entrenamiento en la cola de resultados
        result_queue.put({"message": "Entrenamiento completado con éxito", "model_path": new_model_path})
        os.remove(dataset_path)
        print(f"{Fore.GREEN}[DELETE]{Style.RESET_ALL} Dataset temporal eliminado de {dataset_path}")


    except Exception as e:
        print(f"Error durante el proceso de entrenamiento: {str(e)}")
        # Guardar el error en la cola de resultados
        result_queue.put({"error": f"Error durante el entrenamiento: {str(e)}"})


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


def clear_cache():
    global cache
    cache.clear()
    print(f"{Fore.GREEN}[CLEAR]{Style.RESET_ALL} Caché borrado correctamente")

    cache_file = 'cache.pkl'
    if os.path.exists(cache_file):
        os.remove(cache_file)
        print(f"{Fore.GREEN}[DELETE]{Style.RESET_ALL} Archivo de caché eliminado de {cache_file}")

    return jsonify({"message": "Caché borrado con éxito."}), 200


if __name__ == '__main__':
    app.run(debug=True)
