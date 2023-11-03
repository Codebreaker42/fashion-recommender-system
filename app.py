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
# making resnet50 object where we use imagenet dataset training weights and adding our own layer
model=ResNet50(weights='imagenet',include_top=False, input_shape=(224,224,3))
model.trainable=False
model=Sequential([
    model,
    MaxPooling2D()
])

print(model.summary())
def features_extraction(img_path, model):
    # loading image
    img=image.load_img(img_path,target_size=(224,224))
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
    
    return normalized_result

                                # making a list of all images names
image_names=[]
for img in os.listdir('images'):
    image_names.append(os.path.join('images',img))
# print(len(image_names))

                        # making a features list where all images features stored in a 2d list
features_list=[]
for images in tqdm(image_names):
    features_list.append(features_extraction(images,model))

print(np.array(features_list).shape)

# pickling model
pkl.dump(image_names,open('images_names.pkl','wb'))
pkl.dump(features_list,open('features_list.pkl','wb'))