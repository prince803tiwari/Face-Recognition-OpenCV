
import sys
import streamlit as st

st.title("Deployment Diagnostic")

st.write("Python version:", sys.version)

try:
    import cv2
    st.success(f"OpenCV loaded: {cv2.__version__}")
except Exception as e:
    st.error(f"OpenCV error: {e}")

try:
    import face_recognition
    st.success("face_recognition loaded successfully")
except Exception as e:
    st.error(f"Face recognition error: {e}")

try:
    import av
    st.success(f"PyAV loaded: {av.__version__}")
except Exception as e:
    st.error(f"PyAV error: {e}")
