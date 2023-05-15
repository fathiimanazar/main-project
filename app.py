from flask import Flask, render_template, jsonify
import os
from flask import request


import tensorflow as tf
from tensorflow.keras.utils import load_img,img_to_array

from datetime import datetime


project_dir = os.path.dirname(os.path.abspath(__file__))


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'images'


ALLOWED_EXTENSIONS = {'webp', 'tiff', 'png', 'jpg', 'jpeg'}



# Loads Models
resNetmodel = tf.keras.models.load_model('resnet.h5', compile=False)
vgg19model = tf.keras.models.load_model('vgg19.h5', compile=False)


def predictWithModel(model,img):
    prediction = model.predict(img)
    float_arr = prediction.astype(float)

    # print(round(float_arr[0][0] * 100,2))

    result = "Real"
    if round(float_arr[0][0]) == 0:
        result = "Fake"

    # print(round(float_arr[0][0],2))
    return result    


def preprocessImage(img):
    output_ = img_to_array(img)
    output_ = output_/225
    output_ = output_.reshape((1, output_.shape[0], output_.shape[1], output_.shape[2]))
    return output_


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET'])
def home():
    return render_template("index.html")




@app.route("/predict", methods=["POST"])
def predict():
    if 'photo' not in request.files:
        response = {"status": 500,
                    "status_msg": "File is not uploaded", "message": ""}
        return jsonify(response)


    file = request.files['photo']
    if file.filename == '':
        response = {"status": 500,
                    "status_msg": "No image Uploaded", "message": ""}
        return jsonify(response)


    if file and not allowed_file(file.filename):
        response = {
            "status": 500, "status_msg": "File extension is not permitted", "message": ""}
        return jsonify(response)


    architecture = request.form.get('architecture')
    name = str(datetime.now().microsecond) + str(datetime.now().month) + '-' + str(datetime.now().day) + '.jpg'
    photo = request.files['photo']
    path = os.path.join(app.config['UPLOAD_FOLDER'], name)
    photo.save(path)


    my_image = load_img(path, target_size=(128, 128)) #converts image to array and reshapes
    my_image = preprocessImage(my_image)
  
    denseNet_result = predictWithModel(resNetmodel,my_image) #DenseNet Prediction

    
    my_image = load_img(path, target_size=(224, 224)) #converts image to array and reshapes
    my_image = preprocessImage(my_image)
  
    vgg19_result = predictWithModel(vgg19model,my_image) #VGGFace Prediction
  
   
    
    os.unlink(path)

    response = {"status": 200, "status_msg": [vgg19_result, denseNet_result]}
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)