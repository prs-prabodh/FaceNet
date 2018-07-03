from keras import backend as K
from aligner import align
K.set_image_data_format('channels_first')
import cv2
import os
import numpy as np
import tensorflow as tf
from inception_resnet_v1 import *
from keras.models import load_model, Sequential

#model instantiatiaion
model=Sequential()
FRmodel = faceRecoModel(input_shape=(3, 96, 96))
print('Initialising model...')
FRmodel=load_model('./vitals/goldenModel.h5')

#load database from memory
database = np.load('./vitals/database.npy').item()

#function to sync common database across files
def sync():
    global database
    database=dict({})
    np.save('./vitals/database.npy',database)

#Add identity and face embedding to database
def modify_database(image,name='Unknown'):
    global database
    database[name] = img_to_embedding(image, FRmodel)
    np.save('./vitals/database.npy',database)
    database = np.load('./vitals/database.npy').item()
    return database

#Links database to authentication process
def recognizer(database,image):
    """
    Runs image source driver, if necessary.
    Starts authentication process

    """
    name = identification(image, database, FRmodel)
    print('Identification - ',name)
    return name

#starts authentication process by generating embedding through img_to_encoding()
def identification(image, database, model):
    """
    Implements face recognition for the happy house by finding who is the person on the image_path image.

    image_path -- path to an image
    database -- database containing image encodings along with the name of the person on the image
    model -- Inception model instance in Keras
    min_dist -- the minimum distance between image_path encoding and the encodings from the database
    identity -- string, the name prediction for the person on image_path
    """
    embedding = img_to_embedding(image, model)

    min_dist = 100
    identity = None

    # Loop over the database dictionary's identities and embeddings.
    for (name, db_emb) in database.items():

        # Compute L2 distance between the target "embedding" and the current "emb" from the database.
        dist = np.linalg.norm(db_emb - embedding)

        print('distance for %s is %s' %(name, dist))

        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name
        if dist < min_dist:
            min_dist = dist
            identity = name

    if min_dist > 0.65:
        return 'No match found'
    else:
        return str(identity)

#preprocesses image and passes though neural net to generate embedding
def img_to_embedding(image, model):
    '''
    image is resized to (96 x 96), normalised and fed to neural net input layer
    '''
    image = cv2.resize(image, (96, 96))

    #image channel data reversed
    img = image[...,::-1]

    #normalisation and data format changed to channels_first
    img = np.around(np.transpose(img, (2,0,1))/255.0, decimals=15)

    #preprocess data format
    x_train = np.array([img])

    #feed to neural net input
    embedding = model.predict_on_batch(x_train)

    return embedding

#driver code
if __name__ == "__main__":
    print('Initialised Model')
    print('Reading reference image')
    #replace with full reference image path. example - 'D:/img/image_1.jpg'
    image=cv2.imread('Reference Image Path',1)
    image=align(image)
    cv2.imshow('Labelled Image',image)
    print('Generating embedding')
    database=modify_database(image,'Prabodh')
    print('Embedding generated')
    print('Reading test image')
    #replace with full test image path. example - 'D:/img/image_2.jpg'
    image=cv2.imread('Test Image Path',1)
    image=align(image)
    cv2.imshow('Test Image',image)
    print('Recognition started...')
    recognizer(database,image)
