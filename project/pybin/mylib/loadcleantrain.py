
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from keras import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
import tensorflow as tf
import keras
 
import pickle

from keras.utils.np_utils import to_categorical

from tensorflow.keras.preprocessing import image
import numpy as np
import os
import cv2


from matplotlib import pyplot as plt
import tensorflow




def get_preprocessed_images(images_directory: str, image_size: tuple) -> list:
    stop=100
    images = []
    for img in os.listdir(images_directory):
        img = image.load_img(images_directory+img, target_size=image_size)
        img = image.img_to_array(img)
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
        img = preprocess_input(img)
        images.append(img)

        if stop == 0:
            break
        stop-=1
        print(stop)
    return np.vstack(images)

################################################################################
def load_images():

    SIZE=256
    image_size = (SIZE, SIZE)

    # Load your images and preprocess them.
    cracks_images = get_preprocessed_images("C:/Users/bmadmin/Desktop/Octocat/mohammedbouazzaoui/concrete_inspection_dataset/SDNET2018/X/CX/", image_size)
    non_cracks_images = get_preprocessed_images("C:/Users/bmadmin/Desktop/Octocat/mohammedbouazzaoui/concrete_inspection_dataset/SDNET2018/X/UX/", image_size)

    # Make a numpy array for each of the class labels (one hot encoded).
    cracks_labels = np.tile([1, 0], (cracks_images.shape[0], 1))
    non_cracks_labels = np.tile([0, 1], ( non_cracks_images.shape[0], 1))

    # Concatenate your images and your labels into X and y.
    X = np.concatenate([cracks_images,  non_cracks_images])
    y = np.concatenate([cracks_labels,  non_cracks_labels])

    return X, y
################################################################################
def cleanup_img3(image_array):
    #original_image_array = X[IMAGEPOINT] ### FILL IN ###
    original_image_array = image_array

    img =  cv2.cvtColor(original_image_array, cv2.COLOR_BGR2GRAY)

    #removing small components
    ret, blackAndWhite = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV)

    #dilate
    kernel = np.ones((2,2), np.uint8)
    blackAndWhite = cv2.dilate(blackAndWhite, kernel, iterations=3)

    #decompose
    img2 = blackAndWhite.astype("uint8")
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(img2, None, None, None, 8, cv2.CV_32S)
    sizes = stats[1:, -1] #get CC_STAT_AREA component
    img2 = np.zeros((labels.shape), np.uint8)

    a=list(sizes)

    if a == []:
        return img
    a.sort(reverse=True)
    for i in range(len(a)-1):
        if a[i] > (20 * a[i+1]):
            a[i]=0
        else:
            break
    
    a.sort(reverse=True)

    FILTERDOTUPPER = a[0]

    FILTERDOTLOWER=FILTERDOTUPPER/5

    for i in range(0, nlabels - 1):
        if sizes[i] >= FILTERDOTLOWER and sizes[i] <= FILTERDOTUPPER:   #filter small dotted regions
            img2[labels == i + 1] = 255
            
    resimage = cv2.bitwise_not(img2)
    ret, resimage = cv2.threshold(resimage, 127, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((2,2), np.uint8)
    resimage = cv2.erode(resimage, kernel, iterations=3)
    return (resimage)

###############################################################################################
# merge oiginal with edited image
def merge_images(): 
    IMAGESIZE=256
    XRES=[]
    for image_array in X:
        clean_image=cleanup_img3(image_array)
        XNEW = clean_image/255

        ONE=np.ones((IMAGESIZE,IMAGESIZE))
        ONE3=np.ones((IMAGESIZE,IMAGESIZE,3))
        Z = XNEW
        REVERSE=ONE - Z
        ONE3[:,:,0]=REVERSE
        ONE3[:,:,1]=REVERSE
        ONE3[:,:,2]=REVERSE
        XIM=ONE3 * image_array
        XIM=cv2.resize(XIM, (244,244),interpolation=cv2.INTER_LINEAR)
        XRES.append(XIM)
    XRES = np.array(XRES)
    return XRES



##########################


### 
def split_data():
    

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        XRES, 
        y,
        test_size=0.2, 
        random_state=42, 
        shuffle=True
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, 
        y_train_val,
        test_size=0.2, 
        random_state=42, 
        shuffle=True
    )
    XRES.shape,X_train.shape,X_val.shape,y_train.shape,y_val.shape

    return X_train, y_train, X_val, y_val, X_test, y_test
    ######################################



def model_MobileNetV2(model_save_file : str = './defaultmodelMNV2'):



    # Determine the number of generated samples you want per original sample.
    datagen_batch_size = 16

    # Make a training datagenerator object
    train_datagen = ImageDataGenerator(rotation_range=60, horizontal_flip=True)
    train_generator = train_datagen.flow(X_train, y_train, batch_size=datagen_batch_size)

    # Make a validation datagenerator object
    validation_datagen = ImageDataGenerator(rotation_range=60, horizontal_flip=True)
    validation_generator = validation_datagen.flow(X_val, y_val, batch_size=datagen_batch_size)

    ###############################################################################


    X_train.shape, y_train.shape

    # Import your chosen model!


    # Make a model object. 
    # Make sure you exclude the top part. set the input shape of the model to 224x224 pixels, with 3 color channels.
    #model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
    model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))

    # Freeze the imported layers so they cannot be retrained.
    for layer in model.layers:
        layer.trainable = False
        
    model.summary()

    ##########################################################################


    new_model = Sequential()
    new_model.add(model)
    new_model.add(Flatten())
    new_model.add(Dense(64, activation='relu'))
    new_model.add(Dropout(0.5))
    new_model.add(Dense(2, activation='sigmoid'))

    # Summarize.
    new_model.summary()

    ##########################################################################

    # Compile and fit the model. Use the Adam optimizer and crossentropical loss. 
    # Use the validation data argument during fitting to include your validation data.
    #new_model.compile(optimizer='adam',
    #                  loss='binary_crossentropy',
    #                  metrics=['accuracy'])


    new_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=[tf.keras.metrics.BinaryAccuracy(),
                        tf.keras.metrics.FalseNegatives()])

    ##################################################################

    history = new_model.fit(train_generator,
                            epochs=5,   #10
                            batch_size=8,  #8
                            validation_data=validation_generator
                        )
                        
    new_model.save(model_save_file)
    return history


def load_splitted_data():
    with open("./splitted_data_container", "rb") as fp:   # Unpickling
        container = pickle.load(fp)
    X_train  = container[0]
    y_train  = container[1]
    X_val    = container[2]
    y_val    = container[3]
    X_test   = container[4]
    y_test   = container[5]
    return X_train, y_train, X_val, y_val, X_test, y_test

def save_splitted_data(X_train, y_train, X_val, y_val, X_test, y_test):
    container=[X_train, y_train, X_val, y_val, X_test, y_test]
    with open("./splitted_data_container", "wb") as fp:   #Pickling
        pickle.dump(container, fp)
 


###################################################################3
#def plot_history(history : tensorflow.keras.callbacks.History):
def plot_history(history):
    """ This helper function takes the tensorflow.python.keras.callbacks.History
    that is output from your `fit` method to plot the loss and accuracy of
    the training and validation set.
    """
    fig, axs = plt.subplots(1,3, figsize=(22,6))
    axs[0].plot(history['binary_accuracy'], label='training')
    axs[0].plot(history['val_binary_accuracy'], label='validation')
    axs[0].set(xlabel = 'Epoch', ylabel='Binary accuracy', ylim=[0, 1])
    axs[0].legend(loc='lower right')

    axs[1].plot(history['loss'], label='training')
    axs[1].plot(history['val_loss'], label = 'validation')
    axs[1].set(xlabel = 'Epoch', ylabel='Loss', ylim=[0, 1])
    axs[1].legend(loc='lower right')

    axs[2].plot(history['false_negatives'], label = 'training')
    axs[2].plot(history['val_false_negatives'], label = 'validation')
    axs[2].set(xlabel = 'Epoch', ylabel='false_negatives', ylim=[0, 500])
    axs[2].legend(loc='lower right')
    plt.show()

######################################################################
def predict(model_file, image_file:str):
    image_size = (224, 224)
    new_model = keras.models.load_model(model_file)
   
    original_image = image.load_img(image_file, target_size=image_size)

    # Convert your image pixels to a numpy array of values .
    image_array = image.img_to_array(original_image)

    # Reshape your image dimensions so that the colour channels correspond to what your model expects.
    image_array = image_array.reshape((1, image_array.shape[0], image_array.shape[1], image_array.shape[2]))

    # Preprocess your image with preprocess_input.
    prepared_image = preprocess_input(image_array)

    # Predict the class of your picture.
    prediction = new_model.predict(prepared_image)

    # Print out your result.
    print(prediction)
    if prediction[0][0] > prediction[0][1]:
        predic = "Crack"
        print("#### >>>> CRACK")
    else:
        predic = "No_Crack"
        print("#### >>>> NOCRACK")
    # Show handsome rick.
    plt.figure(figsize=(15,15))

    plt.imshow(original_image)
    plt.show()
    return predic


def predict2(model_file, x):
    image_size = (224, 224)
    new_model = keras.models.load_model(model_file)
   
    original_image = x

    # Convert your image pixels to a numpy array of values .
    #image_array = image.img_to_array(original_image)
    image_array = x

    # Reshape your image dimensions so that the colour channels correspond to what your model expects.
    image_array = image_array.reshape((1, image_array.shape[0], image_array.shape[1], image_array.shape[2]))

    # Preprocess your image with preprocess_input.
    prepared_image = preprocess_input(image_array)

    # Predict the class of your picture.
    prediction = new_model.predict(prepared_image)

    # Print out your result.
    print(prediction)
    if prediction[0][0] > prediction[0][1]:
        predic = "Crack"
        print("#222### >>>> CRACK")
    else:
        predic = "No_Crack"
        print("#2222### >>>> NOCRACK")
    # Show handsome rick.
    plt.figure(figsize=(15,15))

    plt.imshow(original_image)
    plt.show()
    return predic

MODELLING = False
if MODELLING:
    X, y = load_images()
    XRES = merge_images()
    X_train, y_train, X_val, y_val, X_test, y_test = split_data()
    save_splitted_data(X_train, y_train, X_val, y_val, X_test, y_test)
    history = model_MobileNetV2('C:/Users/bmadmin/Desktop/Octocat/mohammedbouazzaoui/concrete_inspection/project/static/defaultmodelMNV2')


    np.save('./defaultmodelMNV2_history2',history.history)
    # Open a file and use dump()
    #with open('C:/Users/bmadmin/Desktop/Octocat/mohammedbouazzaoui/concrete_inspection/project/static/defaultmodelMNV2_history.pkl', 'wb') as file:
    #    pickle.dump(history, file)
    #plot_history(history.history)
else:

    # loading
    print("@@@afte01r")
    historyz=np.load('./defaultmodelMNV2_history2.npy',allow_pickle='TRUE').item()
    #with open('./defaultmodelMNV2_history.pkl', 'rb') as file:
    #    history = pickle.load(file)
    print("@@@afte1r")
    print("@@@@@@@@@@@@",type(historyz),historyz)
    plot_history(historyz)
    print("@@@after2")

    predict(model_file = 'C:/Users/bmadmin/Desktop/Octocat/mohammedbouazzaoui/concrete_inspection/project/static/defaultmodelMNV2', image_file = "C:/Users/bmadmin/Desktop/Octocat/mohammedbouazzaoui/concrete_inspection/project/data/CCCC.jpg")
###################################################################
    print("########################################################")
    X_train, y_train, X_val, y_val, X_test, y_test = load_splitted_data()
    print("@@@@@@@@@@@@@@@@@@@",X_test[0].shape)
    model_file = 'C:/Users/bmadmin/Desktop/Octocat/mohammedbouazzaoui/concrete_inspection/project/static/defaultmodelMNV2'
    model = keras.models.load_model(model_file)
    #prediction = model.predict(X_test[0])

    #predict2('C:/Users/bmadmin/Desktop/Octocat/mohammedbouazzaoui/concrete_inspection/project/static/defaultmodelMNV2', X_test[0])
 



    print("#### >>>>>>>>>>>>y=:",y_test[0])
    # Print out your result.
    #print(prediction)
    if prediction[0][0] > prediction[0][1]:
        predic = "Crack"
        print("#### >>>>>>>>>>>> CRACK")
    else:
        predic = "No_Crack"
        print("#### >>>>>>>>>>>>> NOCRACK")
    # Show handsome rick.
    plt.figure(figsize=(15,15))

    plt.imshow(original_image)
    plt.show()
###########
###
###  END
###


#######################################


''' 
Z-=1
if Z == 1:
    Z=98

# Load your image. Make sure it is loaded in with the right dimensions for your model!
#image_size = (224, 224)
#original_image = image.load_img("C:/Users/bmadmin/Desktop/Octocat/mohammedbouazzaoui/concrete_inspection/project/data/AAAA.jpg", target_size=image_size)

with open('C:/Users/bmadmin/Desktop/Octocat/mohammedbouazzaoui/concrete_inspection_dataset/BACKUP_DATA/Chundata/filelist.txt', 'rt') as f:
    for i in range(Z):
        file=f.readline()
        #print(file)
file=file.strip('\n')

''' 









