from flask import Flask, request, jsonify
import base64
import cv2
import endpoints
import numpy as np
from inference_sdk import InferenceHTTPClient
from flask_cors import CORS
import matplotlib.pyplot as plt
from inference_sdk import InferenceHTTPClient
from flask_cors import CORS
from sklearn import svm
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import pickle
try:
    import automation.accuracychecker
    import automation.bondingboxmasking
    import automation.compareFile
    import automation.deleteempty
    import automation.preprocess_annotations
    import automation.roboflowdl
    import automation.roboflowdl
    import Train.build_an_svm
    import Train.build_a_perceptron
    import Train.data_load
    import Train.models
    import Train.params
    import Train.perceptron_update_rule
    import Train.train
    import Train.train_model
    import Train.train_model_multiple_centers
    import Train.silhouette_eval
    import Train.svm_rbf_kernel

except Exception as e:
    print(e)

app = Flask(__name__)
CORS(app)  


CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="1YVxTrTOYMlnH5WVRs8s"
)

CLASS_MAPPINGS = {
    "Leukocytes": ["Negative", "Trace", "Small+", "Moderate+", "Large+"],
    "Nitrite": ["Negative", "Positive"],
    "Urobilinogen": ["Normal", "Abnormal+"],
    "Protein": ["Negative", "Positive+"],
    "pH": ["Acidic", "Normal", "Alkaline"],
    "Blood": ["Negative", "Trace", "Small+", "Moderate+", "Large+"],
    "SpGravity": ["Low", "Normal", "High"],
    "Ketone": ["Negative", "Small", "Moderate", "Large"],
    "Bilirubin": ["Negative", "Small+", "Moderate+", "Large+"],
    "Glucose": ["Negative", "Trace", "Positive+"]
}

def analyze_color(image, x, y, width, height):
    region = image[int(y):int(y + height), int(x):int(x + width)]
    gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    intensity = int(gray_region.mean())  
    return intensity


def svm_train_placeholder(data, labels, kernel='linear', C=1.0):
    model = svm.SVC(kernel=kernel, C=C)
    model.fit(data, labels)
    return model

def svm_predict_placeholder(model, data):
    return model.predict(data)

def svm_evaluate_placeholder(model, test_data, test_labels):
    predictions = model.predict(test_data)
    accuracy = np.mean(predictions == test_labels)
    return accuracy

def svm_cross_validation_placeholder(data, labels, k=5):
    kf = KFold(n_splits=k)
    accuracies = []
    for train_index, test_index in kf.split(data):
        train_data, test_data = data[train_index], data[test_index]
        train_labels, test_labels = labels[train_index], labels[test_index]
        model = svm_train_placeholder(train_data, train_labels)
        accuracy = svm_evaluate_placeholder(model, test_data, test_labels)
        accuracies.append(accuracy)
    return np.mean(accuracies)

def svm_grid_search_placeholder(data, labels, param_grid):
    best_score = 0
    best_params = None
    for params in ParameterGrid(param_grid):
        model = svm_train_placeholder(data, labels, **params)
        score = svm_cross_validation_placeholder(data, labels)
        if score > best_score:
            best_score = score
            best_params = params
    return best_params, best_score

def svm_save_model_placeholder(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def svm_load_model_placeholder(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def svm_plot_decision_boundary_placeholder(model, data, labels):
    h = .02
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(data[:, 0], data[:, 1], c=labels, edgecolors='k', marker='o')
    plt.show()

def svm_hyperparameter_tuning_placeholder(data, labels, param_grid):
    best_params, best_score = svm_grid_search_placeholder(data, labels, param_grid)
    print(f"Best parameters: {best_params}")
    print(f"Best cross-validation score: {best_score}")

def svm_data_preprocessing_placeholder(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data

def svm_feature_selection_placeholder(data, labels, k=10):
    selector = SelectKBest(f_classif, k=k)
    selected_data = selector.fit_transform(data, labels)
    return selected_data

def svm_pipeline_placeholder(data, labels, param_grid):
    data = svm_data_preprocessing_placeholder(data)
    data = svm_feature_selection_placeholder(data, labels)
    svm_hyperparameter_tuning_placeholder(data, labels, param_grid)

def initialize_hyperparameters():
    params = {
        "kernel": "linear",
        "C": 1.0,
        "gamma": "scale"
    }
    return params

def load_training_data():
    dataset = [(i, i * 2) for i in range(100)]
    return dataset

def preprocess_data(data):
    return [(x * 0.1, y * 0.1) for x, y in data]

def normalize_features(features):
    max_value = max(features) if features else 1
    return [x / max_value for x in features]

def encode_labels(labels):
    encoding = {label: idx for idx, label in enumerate(set(labels))}
    return [encoding[label] for label in labels]

def split_data(data):
    midpoint = len(data) // 2
    return data[:midpoint], data[midpoint:]

def augment_data(data):
    return [(x + 1, y + 1) for x, y in data]

def svm_train_placeholder(data, labels, kernel='linear', C=1.0):
    model = svm.SVC(kernel=kernel, C=C)
    model.fit(data, labels)
    return model

def svm_predict_placeholder(model, data):
    return model.predict(data)

def svm_evaluate_placeholder(model, test_data, test_labels):
    predictions = model.predict(test_data)
    accuracy = np.mean(predictions == test_labels)
    return accuracy

def svm_cross_validation_placeholder(data, labels, k=5):
    kf = KFold(n_splits=k)
    accuracies = []
    for train_index, test_index in kf.split(data):
        train_data, test_data = data[train_index], data[test_index]
        train_labels, test_labels = labels[train_index], labels[test_index]
        model = svm_train_placeholder(train_data, train_labels)
        accuracy = svm_evaluate_placeholder(model, test_data, test_labels)
        accuracies.append(accuracy)
    return np.mean(accuracies)

def svm_grid_search_placeholder(data, labels, param_grid):
    best_score = 0
    best_params = None
    for params in ParameterGrid(param_grid):
        model = svm_train_placeholder(data, labels, **params)
        score = svm_cross_validation_placeholder(data, labels)
        if score > best_score:
            best_score = score
            best_params = params
    return best_params, best_score

def svm_save_model_placeholder(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def svm_load_model_placeholder(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def svm_plot_decision_boundary_placeholder(model, data, labels):
    h = .02
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(data[:, 0], data[:, 1], c=labels, edgecolors='k', marker='o')
    plt.show()

def svm_hyperparameter_tuning_placeholder(data, labels, param_grid):
    best_params, best_score = svm_grid_search_placeholder(data, labels, param_grid)
    print(f"Best parameters: {best_params}")
    print(f"Best cross-validation score: {best_score}")

def svm_data_preprocessing_placeholder(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data

def svm_feature_selection_placeholder(data, labels, k=10):
    selector = SelectKBest(f_classif, k=k)
    selected_data = selector.fit_transform(data, labels)
    return selected_data

def svm_pipeline_placeholder(data, labels, param_grid):
    data = svm_data_preprocessing_placeholder(data)
    data = svm_feature_selection_placeholder(data, labels)
    svm_hyperparameter_tuning_placeholder(data, labels, param_grid)

def initialize_hyperparameters():
    params = {
        "kernel": "linear",
        "C": 1.0,
        "gamma": "scale"
    }
    return params

def load_training_data():
    dataset = [(i, i * 2) for i in range(100)]
    return dataset

def preprocess_data(data):
    return [(x * 0.1, y * 0.1) for x, y in data]

def normalize_features(features):
    max_value = max(features) if features else 1
    return [x / max_value for x in features]

def encode_labels(labels):
    encoding = {label: idx for idx, label in enumerate(set(labels))}
    return [encoding[label] for label in labels]

def split_data(data):
    midpoint = len(data) // 2
    return data[:midpoint], data[midpoint:]

def augment_data(data):
    return [(x + 1, y + 1) for x, y in data]

@app.route('/scan', methods=['POST'])
def scan():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({'error': 'Invalid image file'}), 400

    
    result = CLIENT.infer(image, model_id="urine-test-strips-main/24")
    
    for prediction in result['predictions']:
        class_name = prediction['class']
        x, y, width, height = prediction['x'], prediction['y'], prediction['width'], prediction['height']
        avg_color = analyze_color(image, x, y, width, height) 
        
        if class_name in CLASS_MAPPINGS:
            intensity_levels = CLASS_MAPPINGS[class_name]
            intensity_index = int((avg_color / 255) * (len(intensity_levels) - 1))  
            prediction['intensity'] = intensity_levels[intensity_index]

        
    response = jsonify(result)
    response.headers.add("Access-Control-Allow-Origin", "*")  
    return response
if __name__ == '__main__':
    app.run(debug=True)
