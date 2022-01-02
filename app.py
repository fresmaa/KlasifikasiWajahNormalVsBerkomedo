from flask import Flask, render_template, request

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.applications.inception_v3 import decode_predictions
from tensorflow.keras.applications.inception_v3 import InceptionV3

app = Flask(__name__)
model = InceptionV3()

@app.route('/', methods =['GET'])
def hello_word():
    return render_template('index.html')

@app.route('/', methods =['POST'])
def predict():
    imagefile= request.files['imagefile']
    image_path= "./images/" + imagefile.filename
    imagefile.save(image_path)

    image = load_img(image_path, target_size=(299,299))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    yhat = model.predict(image)
    label = decode_predictions(yhat)
    label = label[0][0]

    classification = '%s (%.2f%%)' % (label[1], label[2]*100)

    return render_template('index.html', prediction = classification)

if __name__ == '__main__':
    app.run(port=3000, debug=True)