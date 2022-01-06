from flask import Flask, render_template, request

import cv2
import numpy as np
from tensorflow import keras


app = Flask(__name__)
model = keras.models.load_model('Komedo-vs-Normal_model (2).h5')

@app.route('/', methods =['GET'])
def hello_word():
    return render_template('index.html')

@app.route('/', methods =['POST'])
def predict():
    imagefile= request.files['imagefile']
    image_path= "static/images/" +  imagefile.filename
    imagefile.save(image_path)

    images = []
    processed_images = []

    image = cv2.imread(image_path)
    if image is not None:
        rescaled = cv2.resize(image, (256, 256))/255.0
        images.append(image)
        processed_images.append(rescaled)

    processed_images = np.array(processed_images)

    prediction = model.predict(processed_images)
    predictions = prediction[0][0]

    if predictions > 0.5:
        title = "Prediksi : Normal \n"
        x_label = "\n Akurasi: {:5.2f}%".format(200*(predictions-0.5))
    else:
        title = "Prediksi : Berkomedo \n"
        x_label = "\n Akurasi: {:5.2f}%".format(200*(0.5-predictions))

    classification = '%s (%s)' % (title, x_label)

    return render_template('index.html', prediction = classification, image_path = image_path)

if __name__ == '__main__':
    app.run(port=3000, debug=True)