import streamlit as st
import pandas as pd
import cv2
from PIL import Image
import numpy as np
import pytesseract

@st.cache
def load_image(img):
        img = Image.open(img)
        return img

st.title("Hello World")

def transform(img,):
        # img = img.convert('LA')
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(img)
        return text

def thresholder_rgb(img):
        l_b = st.slider("LB channel",min_value=0,max_value=255)
        l_g = st.slider("LG channel",min_value=0,max_value=255)
        l_r = st.slider("LR channel",min_value=0,max_value=255)

        u_b = st.slider("UB channel",min_value=0,max_value=255)
        u_g = st.slider("UG channel",min_value=0,max_value=255)
        u_r = st.slider("UR channel",min_value=0,max_value=255)
        b = st.button("ok")
        f=0
        while True:
                lb = np.array([l_b, l_g, l_r])
                ub = np.array([u_b, u_g, u_r])
                mask = cv2.inRange(img, lb, ub)
                res = cv2.bitwise_and(img, img, mask=mask)
                if f==0:
                        st.image(res)
                f = 1
                if b:
                        break
@st.cache
def Preprocessing():
        upload = st.file_uploader("Enter a database...",type=[".csv", ".json"])
        if upload:
                df = pd.read_csv(upload)
                df
                st.text_input()





nav = st.sidebar.radio('Navigation',['Home','OCR'])
if nav == "Home":
        st.write("Welcome to mi casa")


if nav == "OCR":
        upload = st.file_uploader("Choose an image...", type=[".jpg",".png"])
        if upload != None:
                st.image(upload)
                img = load_image(upload)
                b1 = st.button("Load text from your image...")
                if b1:
                        text = transform(img)
                        st.write(text)
