import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import keras

st.set_page_config(page_title="Brain Tumor Classifier", page_icon=":brain:", layout="wide")
st.title('Brain Tumor Classifier')
st.write('')
st.write('This is a python app which classifies a brain MRI into one of the four classes based on the MRI scan images ')
st.write(' Glioma tumor, Meningioma tumor, No tumor or Pituitary tumor')
st.sidebar.title("Select an Image")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
IMAGE_SIZE = 150

from tensorflow.keras.applications import vgg16
model = keras.models.load_model("Brain tumor//Brain_Tumor_Image_Classification_Model.h5")


if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)
    # image = image[:,:,::-1].copy()
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    # plt.imshow(image)
    st.image(image)
    images = image.reshape(1,150,150,3)
    prediction = model.predict(images)
    prediction_class = np.argmax(prediction, axis=1)
    labels = [ 'Glioma tumor', 'Meningioma tumor','No tumor','Pituitary tumor']
    st.write('Prediction over the uploaded image:')
    st.title(labels[prediction_class[0]])
    st.write('Risk score:')
    for i in range(len(labels)):
        st.write(f"{labels[i]}: {prediction[0][i]*100:.2f}%")
