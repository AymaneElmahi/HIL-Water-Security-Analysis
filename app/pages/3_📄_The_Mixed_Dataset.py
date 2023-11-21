import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from source.physical_prep import *
from source.tools import *
import os 
from source.mixed_prep import *
from source.physical_prep import *

st.set_option('deprecation.showPyplotGlobalUse', False)

def choose_model(model_choice):
    if model_choice == "Decision Tree":
        model = DecisionTreeClassifier()
    elif model_choice == "Random Forest":
        model = RandomForestClassifier()
    elif model_choice == "XGBoost":
        model = XGBClassifier()
    elif model_choice == "SVM":
        model = SVC()
    elif model_choice == "MLP":
        model = MLPClassifier()
    elif model_choice == "KNN":
        model = KNeighborsClassifier()
        # If using KNN, scale the features
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        st.error("Invalid model choice. Please choose either 'Decision Tree', 'Random Forest', 'XGBoost', 'KNN', 'SVM', or 'MLP'.")
        
    return model

introduction = "In this section, we will mix the datasets together to see if the physical dataset can improve the performance of the network dataset."
# -------------------------------------------------------------------------------- 
st.title("The Mixed Dataset")
st.write(introduction)

data_overview = "We will start by looking at the overview of the dataset. We will import the two dataset and concatenate them into one dataframe."
# -------------------------------------------------------------------------------- 

st.header("Data Overview")
st.write(data_overview)
root =("dataset/")

merged_df = mix_prep(root)

st.write("Here are the columns of the `Mixed` dataset and their types:")
st.write(merged_df.dtypes)

st.write("As we can see, there are some `object` columns. We will treat them as `categorical` columns.")

st.write("The mixed dataset contains", merged_df.shape[0], "rows and", merged_df.shape[1], "columns.")
st.write("Here is an overview of the mixed dataset:")
st.write(merged_df.head())

st.write("As we already treated both datasets, we will not do any further preprocessing.")
st.write("We will only check the distribution of the labels.")
st.write(merged_df['Label'].value_counts())
st.write(merged_df["Label"].value_counts(normalize=True))

st.write("We can see that the `normal` label is the most frequent one. We will treat that later.")

# --------------------------------------------------------------------------------

st.header("Correlation Analysis")
st.write("We will now plot the correlation matrix of the mixed dataset.")

# check if the image exists in the plots folder
if not os.path.exists('plots'):
    os.makedirs('plots')
    
if not os.path.exists('plots/correlation_matrix_mixed.png'):
    numeric_cols = merged_df.select_dtypes(include=np.number).columns.tolist()
    corr = merged_df[numeric_cols].corr()
    plt.figure(figsize=(30,30))
    sns.heatmap(corr, annot=True, cmap=plt.cm.Reds)
    plt.savefig('plots/correlation_matrix_mixed.png')
    
st.image('plots/correlation_matrix_mixed.png')

st.write("We can see that there are some columns that are highly correlated. We will delete them.")
    
# --------------------------------------------------------------------------------

st.header("Machine Learning")
st.write("We will now train a machine learning model on the mixed dataset.")

import streamlit as st
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from tabulate import tabulate
from sklearn.preprocessing import LabelEncoder

st.subheader("Label Encoding")
categorical_columns = merged_df.select_dtypes(include=['object']).columns.tolist()

st.write("Here are the categorical columns:")
st.write(categorical_columns)
st.write("In this part, we will encode the categorical columns of the mixed dataset.")
label_encoder = LabelEncoder()

# Apply label encoding to categorical columns
for col in categorical_columns:
    merged_df[col] = label_encoder.fit_transform(merged_df[col].astype(str))

# Prepare data for modeling (using Label_n as the target variable)
X = merged_df.drop(['Label', 'Label_n', 'Time','attack_number','attack_type_number'], axis=1)
y = merged_df['Label_n']

# User chooses between Decision Tree and Random Forest
model_choice = st.radio("Choose a Model:", ("Decision Tree", "Random Forest", "XGBoost", "KNN", "SVM", "MLP"))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the selected model
model = choose_model(model_choice)

# Check if the model is selected before proceeding
if 'model' in locals():
    st.write("Here is the model that we will use:")
    st.write(model)
    st.write("We will now train the model.")
    start_time = time.time()
    model.fit(X_train_scaled, y_train)
    end_time = time.time()
    # Evaluate the model
    y_pred = model.predict(X_test_scaled)    
    st.write("The model is trained. We will now evaluate it.")
    # indicate the time elapsed
    st.write("Time elapsed: ", round(end_time - start_time, 2), "seconds")
    st.write("Here are the metrics of the model:")
    plot_metrics(y_test, y_pred, X_test_scaled, model)
    
    st.write("We will now plot the feature importances of the model.")
    plot_feature_importances_plotly(model, X_train.columns)
    
st.write("The model has a weird accuracy of `1.0`.")

st.write("Now let's try to predict the different attacks.")
st.subheader("Predicting the different attacks")

y = merged_df['attack_type_number']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_choice_2 = st.radio("Choose a Model:", ("Decision Tree", "Random Forest", "XGBoost", "KNN", "SVM", "MLP"), index=1)

# Train the selected model
model_2 = choose_model(model_choice_2)

# Check if the model is selected before proceeding
if 'model_2' in locals():
    st.write("Here is the model that we will use:")
    st.write(model_2)
    st.write("We will now train the model.")
    start_time = time.time()
    model_2.fit(X_train_scaled, y_train)
    end_time = time.time()
    # Evaluate the model
    y_pred = model_2.predict(X_test_scaled)    
    st.write("The model is trained. We will now evaluate it.")
    # indicate the time elapsed
    st.write("Time elapsed: ", round(end_time - start_time, 2), "seconds")
    st.write("Here are the metrics of the model:")

    st.success("0 : normal  |  1 : MITM | 2 : physical fault  | 3 : DoS")    
    plot_metrics(y_test, y_pred, X_test_scaled, model_2)
    
    st.write("We will now plot the feature importances of the model.")
    plot_feature_importances_plotly(model_2, X_train.columns)


st.write("The model still has a weird accuracy of `1.0`.")

st.subheader("Dropping the categorical columns")
st.write("We will now drop the categorical columns and see if the accuracy improves.")

# Prepare data for modeling (using Label_n as the target variable)
X = merged_df.drop(['Label', 'Label_n', 'Time','attack_number','attack_type_number'], axis=1)
# drop categorical columns if they exist
X = X.select_dtypes(exclude=['object'])
y = merged_df['Label_n']

# User chooses between Decision Tree and Random Forest
model_choice_3 = st.radio("Choose a Model:", ("Decision Tree", "Random Forest", "XGBoost", "KNN", "SVM", "MLP"), index=2)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the selected model
model_3 = choose_model(model_choice_3)

# Check if the model is selected before proceeding
if 'model_3' in locals():
    st.write("Here is the model that we will use:")
    st.write(model_3)
    st.write("We will now train the model.")
    start_time = time.time()
    model_3.fit(X_train_scaled, y_train)
    end_time = time.time()
    # Evaluate the model
    y_pred = model_3.predict(X_test_scaled)    
    st.write("The model is trained. We will now evaluate it.")
    # indicate the time elapsed
    st.write("Time elapsed: ", round(end_time - start_time, 2), "seconds")
    st.write("Here are the metrics of the model:")
    plot_metrics(y_test, y_pred, X_test_scaled, model_3)
    
    st.write("We will now plot the feature importances of the model.")
    plot_feature_importances_plotly(model_3, X_train.columns)
    
st.write("The model still has a weird accuracy of `1.0`.")

st.subheader("Balancing the dataset")
st.write("We will now balance the dataset and see if the accuracy improves.")

# Prepare data for modeling (using Label_n as the target variable)
X = merged_df.drop(['Label', 'Label_n', 'Time','attack_number','attack_type_number'], axis=1)
# drop categorical columns if they exist
X = X.select_dtypes(exclude=['object'])
y = merged_df['Label_n']

# balance the dataset
from imblearn.over_sampling import SMOTE
oversample = SMOTE()
X, y = oversample.fit_resample(X, y)

# User chooses between Decision Tree and Random Forest
model_choice_4 = st.radio("Choose a Model:", ("Decision Tree", "Random Forest", "XGBoost", "KNN", "SVM", "MLP"), index=3)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the selected model
model_4 = choose_model(model_choice_4)

# Check if the model is selected before proceeding
if 'model_4' in locals():
    st.write("Here is the model that we will use:")
    st.write(model_4)
    st.write("We will now train the model.")
    start_time = time.time()
    model_4.fit(X_train_scaled, y_train)
    end_time = time.time()
    # Evaluate the model
    y_pred = model_4.predict(X_test_scaled)    
    st.write("The model is trained. We will now evaluate it.")
    # indicate the time elapsed
    st.write("Time elapsed: ", round(end_time - start_time, 2), "seconds")
    st.write("Here are the metrics of the model:")
    plot_metrics(y_test, y_pred, X_test_scaled, model_4)
    
    st.write("We will now plot the feature importances of the model.")
    plot_feature_importances_plotly(model_4, X_train.columns)
    
st.write("The model still has a weird accuracy of `1.0`.")

# --------------------------------------------------------------------------------
st.header("Drop the dport column")
st.write("We will now drop the `dport` column and see if the accuracy improves.")

# Prepare data for modeling (using Label_n as the target variable)
X = merged_df.drop(['Label', 'Label_n', 'Time','attack_number','attack_type_number', 'dport'], axis=1)
# drop categorical columns if they exist
X = X.select_dtypes(exclude=['object'])
y = merged_df['Label_n']

# balance the dataset
from imblearn.over_sampling import SMOTE
oversample = SMOTE()
X, y = oversample.fit_resample(X, y)

# User chooses between Decision Tree and Random Forest
model_choice_5 = st.radio("Choose a Model:", ("Decision Tree", "Random Forest", "XGBoost", "KNN", "SVM", "MLP"), index=4)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the selected model
model_5 = choose_model(model_choice_5)

# Check if the model is selected before proceeding
if 'model_5' in locals():
    st.write("Here is the model that we will use:")
    st.write(model_5)
    st.write("We will now train the model.")
    start_time = time.time()
    model_5.fit(X_train_scaled, y_train)
    end_time = time.time()
    # Evaluate the model
    y_pred = model_5.predict(X_test_scaled)    
    st.write("The model is trained. We will now evaluate it.")
    # indicate the time elapsed
    st.write("Time elapsed: ", round(end_time - start_time, 2), "seconds")
    st.write("Here are the metrics of the model:")
    plot_metrics(y_test, y_pred, X_test_scaled, model_5)
    
    st.write("We will now plot the feature importances of the model.")
    plot_feature_importances_plotly(model_5, X_train.columns)






















# --------------------------------------------------------------------------------
Conclusion = "We can see that the mixed dataset does not improve the performance of the network dataset. There is for sure a column that is causing this weird accuracy. We didn't have time to investigate further."

st.header("Conclusion")
st.write(Conclusion)