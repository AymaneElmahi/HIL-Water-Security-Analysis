import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from source.physical_prep import *
from source.tools import *
import os 

st.set_option('deprecation.showPyplotGlobalUse', False)

introduction = "In this section, we will explore the physical dataset."
# -------------------------------------------------------------------------------- 
st.title("The Physical Dataset")
st.write(introduction)


root =("dataset/")
df_physical_raw, df_physical = physical_prep(root)

# write the name of the columns and their types next to it
data_overview = "We will start by looking at the overview of the dataset. We will import the multiple files and concatenate them into one dataframe."
# -------------------------------------------------------------------------------- 

st.header("Data Overview")
st.write(data_overview)

st.write("Here are the columns of the `RAW` dataset and their types:")
st.write(df_physical_raw.dtypes)
st.write("As we can see, the `Time` column is not in the right format. We will convert it to the right format.")
st.write("Also we will convert the `bool` columns to `int` columns.")
st.write("Let's plot the histograms of the `RAW` dataset to see the distribution of the values.")

# plot the histograms of the RAW dataset
# check if the image exists in the plots folder
if not os.path.exists('plots'):
    os.makedirs('plots')
    
if not os.path.exists('plots/histogram.png'):
    df_physical_raw.hist(figsize=(20, 20), bins=50)
    st.pyplot()
    plt.savefig('plots/histogram.png')
    
st.image('plots/histogram.png')

st.write("We can see that some columns have only one unique value. We will delete them.")
st.write("Here are the columns that we kept:")
st.write(df_physical.columns.tolist())
st.write("The prepated physical dataset contains", df_physical.shape[0], "rows and", df_physical.shape[1], "columns.")

st.write(df_physical.head())

st.write("We know that the `label` column contains multiple type of attacks.")
st.write(df_physical_raw['Label'].value_counts())
st.write(df_physical_raw["Label"].value_counts(normalize=True))
st.write("We will then rename the `nomal` label to `normal`, and delete the `scan` rows.")
st.write("We will also create a new column `attack_type_number` that will map the attack type to a number.")

# -------------------------------------------------------------------------------- 

st.header("Time Series Plots")

st.write("Here is the plot of the `Tank_1` column over time for each attack number.")

for attack_number in df_physical['attack_number'].unique():
        plot_feature(df_physical[df_physical['attack_number'] == attack_number], df_physical.columns.to_list()[1], f"{df_physical.columns.to_list()[1]} over time for attack number {attack_number}", streamlit=True)

st.write("You can select a column to plot over time for each attack number.")
# Allow the user to select a column, it cant be the time column
selected_column = st.selectbox("Select a column to plot", df_physical.columns.tolist()[2:])

# Button to trigger the plot update
plot_button = st.button("Plot Selected Column")

# Check if the button is pressed and then plot the selected column
if plot_button:
    for attack_number in df_physical['attack_number'].unique():
        plot_feature(df_physical[df_physical['attack_number'] == attack_number], selected_column, f"{selected_column} over time for attack number {attack_number}", streamlit=True)

# -------------------------------------------------------------------------------- 

st.header("Correlation Analysis")
numeric_columns = df_physical.select_dtypes(include=[np.number]).columns
correlation_matrix = df_physical[numeric_columns].corr()

# save the matrix as a png file
plt.figure(figsize=(20, 16))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)

if not os.path.exists('plots'):
    os.makedirs('plots')

if not os.path.exists('plots/correlation_matrix.png'):
    plt.savefig('plots/correlation_matrix.png')
    
st.image('plots/correlation_matrix.png')

correlation_analysis = """
- The strongest correlation is between Valv_10 and Valv_11 (1.00), which means that these two variables are perfectly correlated. This means that they always move in the same direction and at the same rate.  
- There are also very strong positive correlations between Pump 1 and Pump 2 (0.91), Pump 4 and Pump 5 (0.97), and Flow sensor 1 and Flow sensor 2 (0.99). This means that these pairs of variables are also very closely related.  
- There are some moderate positive correlations between other variables, such as Tank 1 and Tank 2 (0.11), Pump 1 and Flow sensor 1 (0.30), and Valv_10 and Valv_12 (0.22).  
- There are a few negative correlations, but they are all relatively weak. The strongest negative correlation is between Tank 5 and Valv_18 (-0.60), but this is still a moderate correlation.  

Overall, the correlation matrix suggests that the system we are studying is highly interconnected.  

Here are the conclusions we can draw from all this :

- Valv_10 and Valv_11 are likely to be controlled by the same system, as they are perfectly correlated.
- Pump 1 and Pump 2, Pump 4 and Pump 5, and Flow sensor 1 and Flow sensor 2 are also likely to be controlled by the same systems, as they are very strongly correlated.
- Tank 1 and Tank 2 are likely to be connected, but the relationship is more complex, as the correlation is weaker.
- Tank 5 and Valv_18 may be inversely related, but the relationship is also relatively weak.

> **N.B:** It is important to note that correlation does not equal causation. Just because two variables are correlated does not mean that one causes the other. More research would be needed to determine the causal relationships between the variables in this system.
"""

st.write(correlation_analysis)

# --------------------------------------------------------------------------------

st.header("Feature Relationships")

# Allow the user to select a column, it can't be the time column
selected_column_1 = st.selectbox("Select the first column", df_physical.columns.tolist()[1:])
selected_column_2 = st.selectbox("Select the second column", df_physical.columns.tolist()[1:])

# Button to trigger the plot update
plot_button = st.button("Plot Selected Columns", key="plot_button_unique_key")

# Check if the button is pressed and then plot the selected columns
if plot_button:
    plot_feature_feature(selected_column_1, selected_column_2, df_physical)

# --------------------------------------------------------------------------------

st.header("Machine Learning")

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler



# Prepare data for modeling (using Label_n as the target variable)
X = df_physical.drop(['Label', 'Label_n', 'Time','attack_number','attack_type_number'], axis=1)
y = df_physical['Label_n']

# User chooses between Decision Tree and Random Forest
model = st.radio("Choose a Model:", ("Decision Tree", "Random Forest", "XGBoost","KNN","SVM","MLP"))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the selected model
if model == "Decision Tree":
    model = DecisionTreeClassifier()
elif model == "Random Forest":
    model = RandomForestClassifier()
elif model == "XGBoost":
    model = XGBClassifier()
elif model == "SVM":
    model = SVC()
elif model == "MLP":
    model = MLPClassifier()
elif model == "KNN":
    model = KNeighborsClassifier()
    # If using KNN, scale the features
    model.fit(X_train_scaled, y_train)
    
else:
    st.error("Invalid model choice. Please choose either 'Decision Tree', 'Random Forest', 'XGBoost', 'KNN', 'SVM', or 'MLP'.")

# Check if the model is selected before proceeding
from tabulate import tabulate

if 'model' in locals():
    st.write("Here is the model that we will use:")
    st.write(model)
    st.write("We will now train the model.")
    model.fit(X_train_scaled, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test_scaled)    
    st.write("The model is trained. We will now evaluate it.")
    st.write("Here are the metrics of the model:")
    plot_metrics(y_test, y_pred, X_test_scaled, model)

# --------------------------------------------------------------------------------
st.header("Conclusion")
st.write("We have explored the physical dataset. We have seen the overview of the dataset, the time series plots, the correlation analysis, the feature relationships, and the machine learning.")
st.write("We can see that the accuracy is nearly 100% for the Decision Tree and the Random Forest. This is because the dataset is very clean. We will see in the next section that the network dataset is not as clean as the physical dataset. We will then see that the accuracy is not as high as the physical dataset.")