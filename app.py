from flask import Flask, render_template, request
from PIL import Image
import numpy as np
from keras.models import load_model
import tensorflow as tf
import os

app = Flask(__name__)

model = load_model('digit_recognizer.h5')

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict/', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            
            file_path = 'uploaded_image.png'

            file.save(file_path)  
            # Preprocess the uploaded image
            img = Image.open(file_path).convert('L')  
            img = img.resize((28, 28)) 
            img_array = np.array(img)
            img_array = np.invert(img_array)  
            img_array = img_array.reshape(1, 28, 28, 1) 
            img_array = img_array.astype('float32') / 255.0

            # Make predictions using the pre-trained model
            out = model.predict(img_array)
            prediction = np.argmax(out, axis=1)
            response = str(prediction[0])
            return response
        else:
            return "No file uploaded"
    return render_template("index.html")

if __name__ == '__main__':
    app.debug = True
    port = int(os.environ.get("PORT", 5000))
    app.run(host='localhost', port=port)
