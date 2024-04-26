import os
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
model = load_model("cnn.h5", compile=False)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define scene categories
index = ["building", "forest", "glacier", "mountain", "sea", "street"]

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for the about page
@app.route('/about')
def about():
    return render_template('about.html')

# Route for the classification page
@app.route('/classification', methods=['GET', 'POST'])
def classification():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)

        image_file = request.files['image']

        if image_file.filename == '':
            return redirect(request.url)

        if image_file:
            image_path = os.path.join('static/upload', image_file.filename)
            image_file.save(image_path)
            img = preprocess_image(image_path)
            pred = model.predict(img)
            pred_class = np.argmax(pred, axis=1)
            pred_scene = index[pred_class[0]]

            return render_template('classification.html', image_path=image_path, predictions=pred_scene)
    return render_template('classification.html')



if __name__ == '__main__':
    app.run(debug=True)
