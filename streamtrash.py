import streamlit as st
import cv2

st.title("Trashformers - Webcam Feed")

run = st.sidebar.checkbox("RunWebCam", value=False)


cap = cv2.VideoCapture(0)

video_feed = st.empty()

if run:

    #r,image = cap.read()
    

    if not r:
        st.error("error from cap.read")
        break


video_feed.image(image=image, channels="RGB")


cap.release()
