# -*- coding: utf-8 -*-
"""
#########################################################
# MVP for the 'Inspection of concrete structures' project
#########################################################
@author: Bouazzaoui Mohammed
Created on : 20/6/2022

"""

import os
import io
import numpy as np
import pandas as pd
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.preprocessing import LabelEncoder
import autokeras as ak
import tensorflow as tf
from tensorflow import keras
import cv2

# PixelSize and Size of images
SIZE=64
IMGSIZE = (224,224)

# example of loading an image with the Keras API
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.preprocessing.image import save_img

from flask import Flask, render_template, request, session
import os.path
from pybin.mylib.myfunctions import debug

np.random.seed(42)

FILE_model =  "./project/static/ActiveModel"
FILE_imgtopredict = "./project/static/imgtopredict.jpg"

app = Flask( __name__, template_folder= "./pybin/templates/", static_folder =  "./static/" )

# set DEBUG to 'True' to output debug information
DEBUG = False

global modelfile

if os.path.isdir("./project/static"): 
    FILE_model =  "./project/static/ActiveModel"
    FILE_imgtopredict = "./project/static/imgtopredict.jpg"
else:
    FILE_model =  "./static/ActiveModel"
    FILE_imgtopredict = "./static/imgtopredict.jpg"

# get the active model
modelfile = FILE_model
debug(DEBUG, modelfile)
model = tf.keras.models.load_model(modelfile)

# Defining upload folder path
UPLOAD_FOLDER = os.path.join('static', 'uploads')
# # Define allowed files
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
 
# Configure upload folder for Flask application
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
 
# Define secret key to enable session
app.secret_key = 'This is your secret key to utilize session in Flask'


def predict(model_file, image_file:str):
    # Function : will predict using model and an imagefile
    #
    # Input :  
    # Return : render_template 
    
    image_size = IMGSIZE
    new_model = keras.models.load_model(model_file)
   
    original_image = image.load_img(image_file, target_size=image_size)

    # Convert image pixels to a numpy array of values
    image_array = image.img_to_array(original_image)

    # Reshape image dimensions to what the model expects
    image_array = image_array.reshape((1, image_array.shape[0], image_array.shape[1], image_array.shape[2]))

    # Preprocess your image with preprocess_input.
    prepared_image = preprocess_input(image_array)

    # Predict the class of your picture.
    prediction = new_model.predict(prepared_image)

    # Print out result
    if prediction[0][0] > prediction[0][1]:
        predictiontxt = "Cracked"
    else:
        predictiontxt = "Not_Cracked"

    return predictiontxt

@app.route("/uploadfile/", methods=["POST", "GET"])
def index():
    # Function : will render_template
    #
    # Input :  
    # Return : render_template 
    return render_template('index_upload_and_display_image.html')
 
@app.route("/uploadpredict/", methods=["POST", "GET"])
def uploadpredict():
    # Function : will return prediction
    #
    # Input :  
    # Return : render_template 

    if request.method == 'POST':
        # Upload file flask
        uploaded_img = request.files['uploaded-file']
        if str(uploaded_img) == '<FileStorage: \'\' (\'application/octet-stream\')>':
            return render_template('index_upload_and_display_image.html')
        # Extracting uploaded data file name
        img_filename = secure_filename(uploaded_img.filename)
        # Upload file to database (defined uploaded folder in static path)
        uploaded_img.save(os.path.join(app.config['UPLOAD_FOLDER'], img_filename))
        # Storing uploaded file path in flask session
        session['uploaded_img_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
        # Retrieving uploaded file path from session
        img_file_path = session.get('uploaded_img_file_path', None)
        # Display image in Flask application web page

        imagefile = img_file_path

        debug(DEBUG, imagefile)
       

        # load the image
        image = load_img(imagefile)
        save_img(FILE_imgtopredict, image)

        predict_res = predict(FILE_model, FILE_imgtopredict)
        debug(DEBUG, predict)
        return render_template(
            "image_show_prediction.html", predict=predict_res, imagename=img_filename
        )

@app.route("/info/", methods=["POST", "GET"])
def info():
    # Function : will return the model information
    #
    # Input :  
    # Return : render_template 
    
    # get the summary into a string
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + "\n"))
    summary_string = stream.getvalue()
    stream.close()

    info = summary_string.splitlines()
    debug(DEBUG, modelfile)

    return render_template("info.html", info=info)

@app.route("/", methods=["POST", "GET"])
def roott():
    # Function : dummy just show main screen
    #
    # Input :  
    # Return : render_template 
    #
    return render_template("index_upload_and_display_image.html")

#################
# Start the app
#################

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)