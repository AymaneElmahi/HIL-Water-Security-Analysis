import streamlit as st

# Titre de l'application
st.title("Mon Application Streamlit")

# Afficher du texte
st.write("Let's fuck this project")

# Afficher une image
st.image("assets/fig1.jpg")

# Afficher un graphique
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = x ** 2
plt.plot(x, y)
st.pyplot()

# Afficher une entrée utilisateur
user_input = st.text_input("Entrez du texte", "Saisissez quelque chose...")

# Bouton
if st.button("Cliquez ici"):
    st.write("Vous avez cliqué sur le bouton!")

# Sélection
option = st.selectbox("Sélectionnez une option", ["Option 1", "Option 2", "Option 3"])
st.write("Vous avez sélectionné :", option)
