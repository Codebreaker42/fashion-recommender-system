import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from tensorflow.keras.layers import MaxPooling2D
from numpy.linalg import norm
from tensorflow.keras import Sequential
import os
from tqdm import tqdm
import pickle as pkl
from sklearn.neighbors import NearestNeighbors
image_list=pkl.load(open('images_names.pkl','rb'))
features_list=pkl.load(open('features_list.pkl','rb'))

model=ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable=False

model=Sequential([
    model,
    MaxPooling2D()
])
# loading image
img=image.load_img('10000.jpg',target_size=(224,224))
# image to numpy array
img_arr=image.img_to_array(img)
# expending dimensions of image
expand_img_arr=np.expand_dims(img_arr,axis=0)
# making image suitable for resnet50 model
preprocessed_img=preprocess_input(expand_img_arr)
# extracting features (1,2048)
model.predict(preprocessed_img)
# flattening image array from 2d to 1d
result=model.predict(preprocessed_img).flatten()
#normalizing image ranging values from 0 to 1
normalized_result= result/norm(result)

neighbors=NearestNeighbors(n_neighbors=10,metric='cosine')
neighbors.fit(features_list)
distance,indexes=neighbors.kneighbors([normalized_result])
print(indexes)
for idx in indexes[0]:
    print(image_list[idx]) 