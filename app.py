# pip install streamlit opencv-python streamlit-drawable-canvas tensorflow matplotlib

import streamlit as st
import numpy as np
import cv2
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import load_model

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="AI Digit Recognition",
    page_icon="✍🏻",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

.stApp{
background: linear-gradient(to right,#0f2027,#203a43,#2c5364);
}

.title{
text-align:center;
font-size:42px;
font-weight:bold;
color:white;
}

.subtitle{
text-align:center;
font-size:18px;
color:#d9d9d9;
margin-bottom:25px;
}

.prediction{
background:#ffffff15;
padding:20px;
border-radius:15px;
text-align:center;
font-size:30px;
font-weight:bold;
color:#00ff99;
border:2px solid #00ff99;
}

.footer{
text-align:center;
font-size:15px;
color:white;
margin-top:20px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------

model = load_model("digit_recognition_model.keras", compile=False)

# ---------------- HEADER ----------------

st.markdown(
"""
<div class='title'>✍🏻 AI Handwritten Digit Recognition</div>
<div class='subtitle'>
Draw a digit (0-9) inside the box and click Predict
</div>
""",
unsafe_allow_html=True
)

# ---------------- CANVAS ----------------

canvas_result = st_canvas(
    fill_color="rgba(0,0,0,0)",
    stroke_width=14,
    stroke_color="#FFFFFF",
    background_color="#000000",
    width=300,
    height=300,
    drawing_mode="freedraw",
    key="canvas",
)

# ---------------- BUTTON ----------------

if st.button("🔍 Predict Digit", use_container_width=True):

    if canvas_result.image_data is None:
        st.warning("Please draw a digit first.")
    else:

        with st.spinner("AI is predicting..."):

            img = canvas_result.image_data.astype(np.uint8)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            gray = cv2.resize(gray, (28,28))

            gray = gray / 255.0

            gray = gray.reshape(1,784)

            prediction = model.predict(gray)

            digit = np.argmax(prediction)

            confidence = np.max(prediction) * 100

        st.success("Prediction Completed Successfully!")

        st.markdown(
            f"""
            <div class="prediction">
            Predicted Digit <br><br>
            {digit}
            <br><br>
            Confidence : {confidence:.2f}%
            </div>
            """,
            unsafe_allow_html=True
        )

       

