import streamlit as st

import keras
from PIL import Image, ImageOps
import numpy as np
import h5py

st.title("M mode image classsification")
st.header("")
st.text("Upload a Image for image classification")

from img_classification import teachable_machine_classification

uploaded_file = st.file_uploader("Choose an Mmode image ...", type="jpg")
if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        hf = h5py.File('F:/streamlit-trial/VGG16Mmodegood.h5', 'r')
        label = teachable_machine_classification(image, hf)
        if label == 0:
            st.write("The MRI scan has a brain tumor")
        else:
            st.write("The MRI scan is healthy")
