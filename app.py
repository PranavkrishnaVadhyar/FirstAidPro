from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import tensorflow as tf


# Initialize the Flask app
app = Flask(__name__)

# Define the folder where uploaded files will be stored
UPLOAD_FOLDER = 'uploaded_files'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Configure the upload folder for Flask app
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit upload size to 16MB

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'jfif'}

# Function to check if a file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model():
    model_path = 'model.h5'
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

# Predict function
def predict_class(image):
    # Preprocess the image
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    # Load the model
    model = load_model()

    # Predict the class of the image
    predictions = model.predict(image)

    # Get the predicted class index
    predicted_class_index = np.argmax(predictions[0])

    # Map the predicted class index to the corresponding label
    label_mapping = {
        0: 'Degree1',
        1: 'Degree2',
        2: 'Degree3'
    }
    predicted_label = label_mapping[predicted_class_index]

    return predicted_label

# Route for handling file upload and prediction
@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the POST request has a file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    # If the user does not select a file, the browser submits an empty file without a filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Check if the file has an allowed extension
    if file and allowed_file(file.filename):
        # Secure the filename to prevent malicious attacks
        filename = secure_filename(file.filename)
        # Save the file to the upload folder
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # Read the uploaded image
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        uploaded_image = cv2.imread(image_path)

        # Predict the class of the uploaded image
        predicted_class = predict_class(uploaded_image)

        # Respond with JSON containing predicted class
        return jsonify({'predicted_class': predicted_class}), 200

    return jsonify({'error': 'File type not allowed'}), 400

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
