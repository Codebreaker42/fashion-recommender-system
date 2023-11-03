import streamlit as st
import os
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from tensorflow.keras.layers import MaxPooling2D
from numpy.linalg import norm
from tensorflow.keras import Sequential
import numpy as np
import pickle as pkl
st.title('Fashion Recommender System')

features_list=pkl.load(open('features_list.pkl','rb'))
images_list=pkl.load(open('images_names.pkl','rb'))

model=ResNet50(weights='imagenet',include_top=False, input_shape=(224,224,3))
model.trainable=False
model=Sequential([
    model,
    MaxPooling2D()
])
st.write(os.getcwd())
print(model.summary())
def features_extraction(img_path, model):
    st.write(img_path)
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
# step1=file upload and save
# saving file
def save_image(uploaded_image):
        file_location_path=os.path.join('upload_image',uploaded_image.name)
        with open(file_location_path,'wb') as f:
            f.write(uploaded_image.read())

def recommend(features,features_list):
    neighbors=NearestNeighbors(n_neighbors=11,metric='cosine')
    neighbors.fit(features_list)
    distance,indexes=neighbors.kneighbors([features])
    return distance,indexes

uploaded_image=st.file_uploader("upload an image")
if uploaded_image is not None:
    # step1: saving image
    save_image(uploaded_image)
    st.write("file uploaded successfully")
    # displaying image
    st.image(Image.open(uploaded_image))
    # step2= feature extraction
    features=features_extraction(os.path.join("upload_image",uploaded_image.name),model)
    # st.write(features)
    # step3 -> recommendation 
    distances,indexes=recommend(features,features_list)
    # step4-> show recommended images 
    col1,col2,col3,col4,col5=st.columns(5)
    with col1:
        st.image(images_list[indexes[0][1]])
    with col2:
        st.image(images_list[indexes[0][2]])
    with col3:
        st.image(images_list[indexes[0][3]])
    with col4:
        st.image(images_list[indexes[0][4]])
    with col5:
        st.image(images_list[indexes[0][5]])
    with col1:
        st.image(images_list[indexes[0][6]])
    with col2:
        st.image(images_list[indexes[0][7]])
    with col3:
        st.image(images_list[indexes[0][8]])
    with col4:
        st.image(images_list[indexes[0][9]])
    with col5:
        st.image(images_list[indexes[0][10]])
else:
    st.write('Error occur in file upload')