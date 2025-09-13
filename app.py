import streamlit as st
import base64
import pickle
import numpy as np
import pandas as pd

# Load dataset to get feature names
cancer = pd.read_csv("cancer.csv")
features = cancer.drop(['id','diagnosis','Unnamed: 32'], axis=1).columns

if 'page' not in st.session_state:
    st.session_state.page = 'Home'

# Load the saved model
with open("breast_cancer_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load and encode the image
with open("cancer cell2.png", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode()

# Page setup
st.set_page_config(page_title="Breast Cancer Detection", layout="wide")

# Background image CSS
page_bg_img = f"""
<style>
.stApp::before {{
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-image: url("data:image/jpg;base64,{encoded_string}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
    filter: blur(6px);
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Title
title_style = """
<h1 style='text-align: center; color:black; font-size:54px'>
    🩺 Breast Cancer Detection App
</h1>
"""
st.markdown(title_style, unsafe_allow_html=True)

# Navigation buttons styled and closer together, right-aligned
st.markdown("""
<style>
.button-container {
    display: flex;
    justify-content: flex-end;
    gap: 10px;
    margin-right: 20px;
    margin-bottom: 20px;
}
.stButton > button {
    background-color: #6c876b;
    color: white;
    border: none;
    padding: 8px 20px;
    text-align: center;
    text-decoration: none;
    font-size: 16px;
    border-radius: 8px;
    width:110px;
    transition: background-color 0.3s ease;
}
.stButton > button:hover {
    background-color:#8dc922;
    
}
.stForm {
        # border: 2px solid ;
        padding: 20px;
        border-radius: 3px;
        width:1000px;
     
    }

</style>
""", unsafe_allow_html=True)

# Buttons below the title, right-aligned
col1, col2, col3 = st.columns([3, 1, 2])
with col3:
    cols = st.columns([1, 1, 1])
    with cols[0]:
        if st.button("🏠 Home"):
            st.session_state.page = "Home"
    with cols[1]:
        if st.button("🔑 Login"):
            st.session_state.page = "Login"
    with cols[2]:
        if st.button("ℹ️ About"):
            st.session_state.page = "About"

# Display page content based on the current selection
if st.session_state.page == "Home":
    st.sidebar.header("Input Parameters")

    def user_input_features():
        data = {}
        for feature in features:
            min_val = float(cancer[feature].min())
            max_val = float(cancer[feature].max())
            mean_val = float(cancer[feature].mean())
            data[feature] = st.sidebar.slider(feature, min_val, max_val, mean_val)
        return pd.DataFrame(data, index=[0])

    input_df = user_input_features()

    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    result = "Benign ✅" if prediction[0] == "B" else "Malignant ✅"
    if prediction[0] == "B":
       result_color = "#9a7e27"   # Color for Benign
    else:
       result_color = "#9a4a27"

    st.markdown("<h2 style='color: #277d9a;'>Prediction Result</h2>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background-color: #f9f9f9; padding: 10px; border-radius: 8px;">
        <p style="color: {result_color};font-size: 18px;">The diagnosis is <b>{result}</b></p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<h2 style='color: #277d9a;'>Prediction Probability</h2>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background-color: #eef2f3; padding: 10px; border-radius: 8px;">
        <p style="color: #9a7e27; font-size: 18px;">Probability of being Benign: <b>{prediction_proba[0][0]*100:.2f}%</b></p>
        <p style="color: #9a4a27; font-size: 18px;">Probability of being Malignant: <b>{prediction_proba[0][1]*100:.2f}%</b></p>
    </div>
    """, unsafe_allow_html=True)

    if st.checkbox("Show Input Parameters"):
        st.write(input_df)

elif st.session_state.page == "Login":
    st.markdown("<h2 style='color: #277d9a;'>🔑 Login Page</h2>", unsafe_allow_html=True)
    # st.subheader("🔑 Login Page")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        if submitted:
            if username == "admin" and password == "password":
                st.success("Logged in successfully!")
            else:
                st.error("Invalid username or password.")

elif st.session_state.page == "About":
    st.markdown("<h2 style='color: #277d9a;'>ℹ️ About Page</h2>", unsafe_allow_html=True)
    # st.subheader("ℹ️ About Page")
    st.markdown("""
        <span style="color: #413b38; font-size: 18px;">
•  This Breast Cancer Detection App is built using **Streamlit**.
•  It allows users to input various medical parameters and predicts whether the tumor is **benign** or **malignant** using a logistic regression model.

•  Breast cancer occurs when abnormal cells in the breast grow uncontrollably. Early detection can improve treatment outcomes, and this app is designed to help you learn more about it.

•  A **benign tumor** is a growth that is **not cancerous**. It usually grows slowly and does not spread to other parts of the body. Benign tumors are often less harmful, but it’s still important to consult a doctor for proper evaluation.

•  A **malignant tumor** is **cancerous**. It may grow rapidly and spread to nearby tissues or other parts of the body. Malignant tumors require timely medical attention and treatment by healthcare professionals.
</span>
""", unsafe_allow_html=True)

    st.markdown("<h2 style='color: #277d9a;'>📊 How This App Works</h2>", unsafe_allow_html=True)
    st.markdown("""
        <span style="color: #413b38; font-size: 18px;">
                
    - You input various medical measurements related to breast tumors.
                
    - Our machine learning model analyzes these inputs to predict if the tumor is likely benign or malignant.
    
    - The result is presented with the probability of each outcome.
        </span>
""", unsafe_allow_html=True)


    st.markdown("<h2 style='color: #277d9a;'>💡 Important Disclaimer</h2>", unsafe_allow_html=True)
    st.markdown("""
        <span style="color: #413b38; font-size: 18px;">
This app is for educational and awareness purposes only. It is not a replacement for professional medical advice or diagnosis. Always consult a healthcare provider for medical concerns.
</span>
""", unsafe_allow_html=True)


# Back to Home button if not on Home page
# if st.session_state.page != "Home":
#     if st.button("🏠 Back to Home"):
#         st.session_state.page = "Home"
