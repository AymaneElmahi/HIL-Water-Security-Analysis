import streamlit as st

st.set_page_config(page_title="Projet Protection des Données", 
                   page_icon="🧊", layout="wide",
                     initial_sidebar_state="expanded")

st.title("Projet Protection des Données")
st.header("Data Analysis")
st.sidebar.success("Select What Page To Look At")
st.markdown("This project is a visualisation of the data collected from the Hardware-in-the-loop Water Distribution Testbed dataset for cyber-physical security testing. We will run some analysis on the data, and try to predict the anomaly in the system using machine learning. We will also make a streamlit app to visualize the data and the results of our analysis.")

st.write("You should first check the README.md file to understand the project and where the data comes from. Then you can select the page you want to look at in the sidebar on the left.")
