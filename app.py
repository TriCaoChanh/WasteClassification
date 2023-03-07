import streamlit as st
import numpy as np
import cv2
import av
import os
from tensorflow import keras
from streamlit_webrtc import webrtc_streamer


@st.cache_resource
def load_model():
    model = keras.models.load_model('./waste_classifier')
    return model

def callback_predict(frame):    
    pic = frame.to_ndarray(format='rgb24')
    copy = cv2.resize(pic, (128,128)) / 255
    pred = model.predict(copy.reshape(1,128,128,3), verbose=0)
    pred_class = 'Recyclable' if pred >= THRESHOLD else 'Organic'
    proba = pred if pred >= THRESHOLD else 1-pred
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 50)
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2
    pic = cv2.putText(pic, '%s: %.2f%%' % (pred_class, proba[0][0] * 100), org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
    return av.VideoFrame.from_ndarray(pic)

def random_images():
    path_o = './DATA/O'
    file = np.random.choice(os.listdir(path_o))
    pic_o = cv2.imread(os.path.join(path_o,file))

    path_r = './DATA/R'
    file = np.random.choice(os.listdir(path_r))
    pic_r = cv2.imread(os.path.join(path_r,file))
    
    pic_o = cv2.cvtColor(pic_o, cv2.COLOR_BGR2RGB)
    pic_r = cv2.cvtColor(pic_r, cv2.COLOR_BGR2RGB)
    copy_o = cv2.resize(pic_o, (128,128)) / 255
    copy_r = cv2.resize(pic_r, (128,128)) / 255

    pred_o = model.predict(copy_o.reshape(1,128,128,3), verbose=0)
    pred_r = model.predict(copy_r.reshape(1,128,128,3), verbose=0)

    col1, col2 = st.columns(2)
    with col1:
        pred_class = 'Recyclable' if pred_o >= THRESHOLD else 'Organic'
        proba = pred_o if pred_o >= THRESHOLD else 1-pred_o
        st.markdown('%s: %.2f%%' % (pred_class, proba[0][0] * 100))

        pic_o = cv2.resize(pic_o, (512,512))
        st.image(pic_o, caption="Organic")

    with col2:
        pred_class = 'Recyclable' if pred_r >= THRESHOLD else 'Organic'
        proba = pred_r if pred_r >= THRESHOLD else 1-pred_r
        st.markdown('%s: %.2f%%' % (pred_class, proba[0][0] * 100))

        pic_r = cv2.resize(pic_r, (512,512))
        st.image(pic_r, caption="Recyclable")

model = load_model()
THRESHOLD = 0.3

st.title('Waste Classification Web App')
st.markdown('Portable AI classifier can be integrated into small-size cameras with accuracy over 90% identifying waste images.')
st.markdown('---------------------------------------------------------------')

    

mode = st.sidebar.selectbox('Mode', ['Random Images', 'Import Images', 'Camera'])
if mode == 'Random Images':
    if st.button('Generate'):
        random_images()
elif mode == 'Import Images':
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        pic = cv2.imdecode(file_bytes, 1)
        pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
        copy = cv2.resize(pic, (128,128)) / 255
        pred = model.predict(copy.reshape(1,128,128,3), verbose=0)

        pred_class = 'Recyclable' if pred >= THRESHOLD else 'Organic'
        proba = pred if pred >= THRESHOLD else 1-pred
        st.markdown('%s: %.2f%%' % (pred_class, proba[0][0] * 100))

        pic = cv2.resize(pic, (384,384))
        st.image(pic, caption="Uploaded Image")
elif mode == 'Camera':
    webrtc_streamer(key="example", video_frame_callback=callback_predict)
else:
    st.error("Please Select the Mode")

st.markdown('---------------------------------------------------------------')
st.write('Created by Tri Cao Chanh 2023')
