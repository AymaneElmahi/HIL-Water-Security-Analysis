import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from source.physical_prep import *
from source.tools import *
import os 
from source.network_prep import *

st.set_option('deprecation.showPyplotGlobalUse', False)

def count_unique_combinations(df):
    # Create a new column for combined mac_src and ip_src
    df['mac_ip_src'] = df['mac_s'] + '<->' + df['ip_s']

    # Create a new column for combined mac_dst and ip_dst
    df['mac_ip_dst'] = df['mac_d'] + '<->' + df['ip_d']

    # unique combinations for mac_src and ip_src
    unique_mac_ip_src = df[['mac_ip_src']].drop_duplicates()
    for index, row in unique_mac_ip_src.iterrows():
        st.write(f"MAC_IP_SRC: {row[0]}")

    st.write('------------------dst------------------')
    
    # unique combinations for mac_dst and ip_dst
    unique_mac_ip_dst = df[['mac_ip_dst']].drop_duplicates()
    for index, row in unique_mac_ip_dst.iterrows():
        st.write(f"MAC_IP_DST: {row[0]}")
        
def comprehensive_correlation_analysis(data):
    """
    Perform comprehensive correlation analysis for all pairs of continuous variables in the dataset.

    Parameters:
    - data: Pandas DataFrame containing the dataset.

    Returns:
    - correlation_matrix: Pandas DataFrame representing the correlation matrix.
    """
    # Select only continuous variables
    continuous_variables = data.select_dtypes(include='number')

    # Compute the correlation matrix
    correlation_matrix = continuous_variables.corr()

    # Create a heatmap
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    
    # Save the plot to a buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    # Display the plot
    st.subheader("Correlation Matrix Heatmap")
    st.image(buffer.getvalue(), width=600)  # Adjust the width
    # Close the plot to release resources
    plt.close()

    return correlation_matrix

def downsample_data(data, label_column, target_percentage):
    """
    Downsample the data while conserving the same percentage of each label.

    Parameters:
    - data: Pandas DataFrame containing the dataset.
    - label_column: Name of the column containing the labels.
    - target_percentage: Target percentage to conserve for each label.

    Returns:
    - downsampled_data: Pandas DataFrame with downsampled data.
    """
    # Calculate the target count for each label
    target_counts = (data[label_column].value_counts() * target_percentage).round().astype(int)

    # Downsample each label
    downsampled_data = pd.DataFrame()
    for label, count in target_counts.items():
        label_data = data[data[label_column] == label]
        downsampled_label = label_data.sample(count, replace=False)  # Change replace to True for sampling with replacement
        downsampled_data = pd.concat([downsampled_data, downsampled_label])

    # order the data by time
    downsampled_data = downsampled_data.sort_values(by=['Time'])

    return downsampled_data

def downsample_date2(data, step):
    """
    Downsample the data by step 

    return downsampled_data
    """
    downsampled_data = pd.DataFrame()
    for i in range(0, len(data), step):
        downsampled_data = pd.concat([downsampled_data, data.iloc[i:i+1]])
    return downsampled_data



def present_dataset(df):
    
    st.write("There are {} rows and {} columns in the dataset".format(df.shape[0], df.shape[1]))
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    # Display the information string
    st.text(info_str)
    
    st.write("The percentage of missing values in each column : ")
    st.write(df.isnull().sum()/df.shape[0]*100)
    st.write("For each column name eliminate the spaces :")
    df.columns = df.columns.str.replace(' ', '')
    st.write("There are {} unique source port in sport".format(df['sport'].nunique()),"over {} ports".format(df.shape[0]))
    st.write("There are {} unique destination port in dport".format(df['dport'].nunique()),"over {} ports".format(df.shape[0]))
    df['proto'].unique()
    st.write("The number of protocol is ``2 = [ modbus and tcp ]``, note that the number of o protocol is well defined and any change in the number of o protocol indicates a change in the network or a malicious activity")
    # for i in df.columns[1:]:
    #     st.write("there are {} unique values in {} column".format(df[i].nunique(), i))
    # count_unique_combinations(df)
    st.write("Effectivly in the network normal dataset there is no change in the mapping of the mac address and the ip address any change could indicate a a MITM attack.")
    df['Time'] = pd.to_datetime(df['Time'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
    df = df.dropna(subset=['Time'])
    time_series_df = df.groupby('Time').size()
    st.write(df.columns.tolist())
    st.write(df.head())
    correlation_matrix = comprehensive_correlation_analysis(df)
    
    st.write("Now we will merge all the dataset and downsample the dataset to 0.01/100 of the original dataset")
    st.write(df['label'].value_counts())
    

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
    
    

introduction = "In this section, we will explore the network dataset."
# -------------------------------------------------------------------------------- 
st.title("The Network Dataset")
st.write(introduction)

# --------------------------------------------------------------------------------
st.header("Data Overview")
st.write("We will start by looking at the overview of the dataset. Due to the size of the dataset, we will first explore one of the datasets for example.")

root =("dataset/")
n = st.selectbox("Select the attack dataset number", [0,1, 2, 3, 4], key='attack_number')

# make a download button on the sidebar
df = pd.DataFrame()

download = st.button('Download the dataset', key='download')

if download:
    if n == 0 : 
        df = pd.read_csv(os.path.join(root, "Network datatset/csv/normal.csv"))
    else :
        df = pd.read_csv(os.path.join(root, "Network datatset/csv/attack_" + str(n) + ".csv"))
    
    # add column with the attack number
    df['attack_number'] = n
    st.write(df.head())
    present_dataset(df)
    

# check if the downsampled dataset exists
# if not os.path.exists('dataset/Network datatset/csv/network_downsampled.csv'):
#     df_2 = network_prep(root)
#     st.write(df.head())
#     downsampled_data = downsample_data(df_2, 'label', 0.01)
#     downsampled_data.to_csv('dataset/Network datatset/csv/network_downsampled.csv', index=False)
#     st.write("Successfully downsampled the dataset")
# else:

st.write("The downsampled dataset already exists")
downsampled_data = pd.read_csv('dataset/Network datatset/csv/network_downsampled.csv')

st.write(downsampled_data.Label.value_counts())
# rename the label column to Label and label_n to Label_n
downsampled_data = downsampled_data.rename(columns={'label': 'Label', 'label_n': 'Label_n'})
    
st.success("Successfully downsampled the dataset")
    
st.header("Time Series Plots")


# use plot_feature function to plot the time series plot for each column (only numerical columns with int64 or float64 data types)
numerical_columns = downsampled_data.select_dtypes(include=['int64', 'float64']).columns.to_list()

st.write(numerical_columns)


for attack_number in downsampled_data['attack_number'].unique():
    plot_feature(downsampled_data[downsampled_data['attack_number'] == attack_number], numerical_columns[0], f"{numerical_columns[0]} over time for attack number {attack_number}", streamlit=True)

selected_column = st.selectbox("Select a column to plot", numerical_columns, key='selected_column')

plot_button = st.button('Plot the selected column')
if plot_button:
    for attack_number in downsampled_data['Label_n'].unique():
        plot_feature(downsampled_data[downsampled_data['attack_number'] == attack_number], selected_column, f"{selected_column} over time for attack number {attack_number}", streamlit=True)

    
st.header("Machine Learning")

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

st.subheader("Dropping the categorical columns")
st.write("We will now drop the categorical columns and see if the accuracy improves.")

# Prepare data for modeling (using Label_n as the target variable)
X = downsampled_data.drop(['Label', 'Label_n', 'Time','attack_number','attack_type_number'], axis=1)
# drop categorical columns if they exist
X = X.select_dtypes(exclude=['object'])
# remap the label column to numbers
y = downsampled_data['Label_n']

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

st.write("Now let's try to predict the different attacks.")
st.subheader("Predicting the different attacks")

y = downsampled_data['attack_type_number']
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