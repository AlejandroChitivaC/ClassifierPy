# Hazardous Content Classifier

This project implements a classifier to predict whether a content is hazardous or not. It uses a machine learning model trained on a dataset of text with binary labels (`0` for non-hazardous content and `1` for hazardous content). The model combines TF-IDF vectorization with an SVM classifier to make predictions.

## Features

- **Automatic Training**: The system supports automatic model training using a new dataset, combining it with the original dataset to improve accuracy.
- **Retraining**: The model can be retrained with new data to adapt to new trends or changes in data.
- **Content Prediction**: The API allows you to submit content for predictions on whether it is hazardous or not.
- **Prediction Cache**: It uses a caching system to store previous predictions and accelerate the classification process for similar content.
- **Model Statistics Report**: Generates model statistics reports such as accuracy, recall, and F1-Score to evaluate model performance.
- **Automatic Cleanup**: After training is complete, the system automatically deletes temporary data files to free up space.

## Project Structure

- **`classifier.py`**: Main file containing the Flask API for handling training and classification requests.
- **`datasets/`**: Directory containing datasets for training and storing results.
- **`modelsft/`**: Directory storing trained models.
- **`cache.pkl`**: Cache file storing previous predictions to improve performance.

## How to Use

1. **Training**:
    - Send a POST request to `/train` with a CSV file containing the new dataset.
    - The system will combine the dataset with the original one and retrain the model.
2. **Prediction**:
    - Send a POST request to `/mlclassifier` with a list of contents to predict.
    - You will receive a JSON response with the contents and their corresponding predictions.
3. **Model Verification**:
    - Send a GET request to `/getModelName` to obtain information about the most recent trained model, (Version and model name).

## Requirements

- Python 3.11
- Flask
- Pandas
- scikit-learn
- colorama (For information messages on terminal)
- tqdm
- joblib
- pickle
- torch (optional, if using PyTorch)

## Installation

Install dependencies using pip:

```bash
pip install -r requirements.txt
```

## Running the App

Run the Flask application:

```bash
python classifier.py
```

The application will run on `http://127.0.0.1:5000/`.

## Notes

- Ensure you have a `requirements.txt` file that contains the necessary dependencies.
- Keep datasets updated to improve model performance.




## Authors

- [@AlejandroChitivaC](https://www.github.com/AlejandroChitivaC)

