# -*- coding: utf-8 -*-
"""

MVP for the 'Inspection of concrete structures' project
#######################################################
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

from keras.utils.np_utils import to_categorical

from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.preprocessing import LabelEncoder
import autokeras as ak

import tensorflow as tf
from tensorflow import keras
import cv2

SIZE=16
# example of loading an image with the Keras API
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.preprocessing.image import save_img

from flask import Flask, render_template, request

from pybin.mylib.myfunctions import debug

np.random.seed(42)

app = Flask(__name__, template_folder="/project/pybin/templates")

# set DEBUG to 'True' to output debug information
DEBUG = False

global modelfile

# get the active model
#modelfile = "./project/pybin/models/activemodel_best_trained2"
#modelfile = "C:/Users/bmadmin/Desktop/Octocat/mohammedbouazzaoui/concrete_inspection/project/activemodel"
modelfile = "/project/pybin/models/activemodel_best_trained2"
debug(DEBUG, modelfile)
model = tf.keras.models.load_model(modelfile)
debug(DEBUG, modelfile)


def predictimage(img, model):
    # Function will return a prediction
    #
    # Input : image, model
    # Return : prediction string
    #
    #Convert dataframe column of images into numpy array
    X = np.asarray(img.tolist())
    X = X/255. # Scale values
    Ximg=np.reshape(X, (-1, SIZE, SIZE, 3)) 
    #
    rezzult=model.predict(Ximg)
    print("********************",rezzult)

    # print result
    #print(result)
   

    # report details about the image
    #debug(DEBUG, type(img))
    #debug(DEBUG, img.format)
    #debug(DEBUG, img.mode)
    #debug(DEBUG, img.size)

    # convert to numpy array
    img = img_to_array(img)
    #print(img)
    print("===========================================")
    img = cv2.resize(img, (SIZE, SIZE))
    #print(img)
    print("-----------------------------------------------")
    # new image has to be rescaled/reshaped as in training model 
    img = img / 255.0
    img = np.reshape(img, (-1, SIZE, SIZE, 3))
    #print(img)
    # predict
    result = model.predict(img)
    #result = result[0]
    debug(DEBUG, result)
    print("********************",rezzult)
    # binary classification :crack or not crack:
    #seven = {
    #    "crack": "crack",
    #    "nocrack": "nocrack",
    #}

    #le = LabelEncoder()
    #le.fit(list(seven.keys()))

    # transform result to %
    #som = sum(result)
    #reslist = []
    #for i in range(2):
    #    reslist.append([int(100 * result[i] / som), le.classes_[i]])
    #reslist.sort()
    #debug(DEBUG, reslist)

    #
    return f"Predicted type : {rezzult}  with {result}% accuracy."
   


@app.route("/info/", methods=["POST", "GET"])
def info():
    # Function will return the model information
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


@app.route("/image_show_prediction/", methods=["POST", "GET"])
def image_show_prediction():
    # Function  
    #
    # Input :  
    # Return : render_template  
    #
    debug(DEBUG, "image_predict")
    form_data = request.form

    debug(DEBUG, form_data)

    return render_template("image_show_prediction.html")


@app.route("/image_load/", methods=["POST", "GET"])
def image_load():
    # Function  
    #
    # Input :  
    # Return : render_template 
    #
    return render_template("image_load.html")


@app.route("/image_predict/", methods=["POST", "GET"])
def image_predict():
    # Function does prediction
    #
    # Input :  
    # Return : render_template 
    #
 
    form_data = request.form  # get image filename
    debug(DEBUG, form_data)
    imagename = form_data["myfile"]
    if imagename == "":
        return render_template("image_load.html")

    #imagefile = "./project/data/" + imagename
    imagefile = "/project/data/" + imagename

    debug(DEBUG, imagefile)
    #imagefile = "../concrete_inspection_dataset/SDNET2018/P/CP/001-100.jpg"
    # img = np.asarray(Image.open(pic).resize((SIZE,SIZE)))

    # load the image
    img = Image.open(imagefile)
    #image = load_img(imagefile)
    #save_img("./project/static/imgtopredict.jpg", img)
    save_img("/project/static/imgtopredict.jpg", img)

    img = np.asarray(img.resize((SIZE,SIZE)))
    

    predict = predictimage(img, model)


    # binary classification :crack or not crack:
    seven = {
        "crack": "crack",
        "nocrack": "nocrack",
    }

    debug(DEBUG, predict)
    return render_template(
        "image_show_prediction.html", predict=predict, imagename=imagename
    )




@app.route("/", methods=["POST", "GET"])
def roott():
    # Function : dummy just show main screen
    #
    # Input :  
    # Return : render_template 
    #
    return render_template("image_load.html")


#######################################
# Start the local webserver
#######################################

#app.run(host="localhost", port=5000)
'''
if __name__ == "__main__":
    app.secret_key = "super_secret_key"
    app.config["SESSION_TYPE"] = "filesystem"
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, threaded=True, debug=True)
'''

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)