import streamlit as st
import numpy as np
from keras.models import load_model
import cv2

savedModel = load_model('fmdModel.h5')
image_size = 150
new_image =[]
md = "<div style='text-align: center;'>Just Give It A Try</div>"

print(savedModel.summary())


st.title('Face Mask Detection')

st.subheader("Upload An Image To Classify")

uploaded_file = st.file_uploader("Choose an Image File")

clicked = st.button("Classify")



if uploaded_file is not None:

    #To read file as bytes:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    new_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)

    new_image = cv2.resize(new_image, (image_size, image_size))



if clicked and new_image != []:

    prediction = savedModel.predict(np.array([new_image]))
    pred = prediction[0][0]

    if pred > 0.6:    
        md = "<h5 style='text-align: center;color: #0F0;'>With Mask</h5>"
    else:
        md = "<h5 style='text-align: center;color: #F00;'>Without Mask</h5>"


st.markdown(md, unsafe_allow_html=True)
