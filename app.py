from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import tensorflow as tf
import openai
from pyngrok import ngrok


NGROK_AUTH_TOKEN='2f4v0QaK1n3fh1x4FkU1C0dOJmV_DLshzscPCsbUXBW57g5b'
app = Flask(__name__)
port = "5000"


ngrok.set_auth_token(NGROK_AUTH_TOKEN)
public_url = ngrok.connect(port).public_url

print("ngrok tunnel " + public_url +  "-> http://127.0.0.1:" + port)

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

def load_degree_model():
    model_path = 'model.h5'
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

def load_type_model():
    model_path = 'model2.h5'  # Path to the wound type prediction model
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

'''
def load_model_treatment():
    # Load the OpenAI API key from environment variables
    openai_api_key = os.getenv('OPENAI_API_KEY')
    openai.api_key = openai_api_key

'''
# Predict function for wound type and degree
def predict_class_and_wound(image):
    # Preprocess the image for degree prediction model
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    # Load the degree prediction model
    model = load_degree_model()

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
    predicted_degree = label_mapping[predicted_class_index]

        # Load the degree prediction model
    model = load_type_model()

    # Predict the class of the image
    predictions2 = model.predict(image)

    # Get the predicted class index
    predicted_class_index2 = np.argmax(predictions2[0])

    label_mapping2 = {
        0: 'Abrasion',
        1: 'Burn',
        2: 'Cut',
        3: 'Laceration'
    }
    predicted_type = label_mapping2[predicted_class_index2]

    # Predict the wound type
    #predicted_wound_type = predict_wound(image)

    return predicted_degree, predicted_type

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

        # Predict the class and wound type of the uploaded image
        predicted_degree, predicted_type  = predict_class_and_wound(uploaded_image)

        # Initialize OpenAI API
        #load_model_treatment()
        
        # Request treatment suggestion from OpenAI API
        prompt = f"Type of Wound: {predicted_type}\nDegree: {predicted_degree}\nRequesting treatment suggestions for this wound."
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=150
        )

        # Extract the treatment suggestion from OpenAI API response
        treatment_suggestion = response.choices[0].text.strip()
        
        # Respond with JSON containing predicted class, predicted wound type, and treatment suggestion
        return jsonify({
            'predicted_class': predicted_degree,
            'predicted_wound_type': predicted_type,
            "treatment_steps": treatment_suggestion
        }), 200

    return jsonify({'error': 'File type not allowed'}), 400

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
