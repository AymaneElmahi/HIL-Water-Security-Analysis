import sys
sys.path.append("source/")


import os
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns


def network_prep(root):
    df_network_normal = pd.read_csv(os.path.join(root, "Network datatset/csv/normal.csv"))
    df_network_att_1 = pd.read_csv(os.path.join(root, "Network datatset/csv/attack_1.csv"))
    df_network_att_2 = pd.read_csv(os.path.join(root, "Network datatset/csv/attack_2.csv"))
    df_network_att_3 = pd.read_csv(os.path.join(root, "Network datatset/csv/attack_3.csv"))
    df_network_att_4 = pd.read_csv(os.path.join(root, "Network datatset/csv/attack_4.csv"))
    
    df_network_normal["attack"] = 0
    df_network_att_1["attack"] = 1
    df_network_att_2["attack"] = 2
    df_network_att_3["attack"] = 3
    df_network_att_4["attack"] = 4
    
    for df in [df_network_normal, df_network_att_1, df_network_att_2, df_network_att_3, df_network_att_4]:
        df.columns = df_network_normal.columns
    
    df_network = pd.concat([df_network_normal, df_network_att_1, df_network_att_2, df_network_att_3, df_network_att_4])
    df_network['attack_type_number'] = df_network['label'].map({'normal': 0, 'MITM': 1, 'physical fault': 2, 'DoS': 3, 'anomaly': 4, 'scan': 5})    

    df_network = df_network[df_network.label != 'anomaly']
    df_network = df_network[df_network.label != 'scan']

    df_network = df_network.dropna()

    df_network['attack_type_number'] = df_network['attack_type_number'].astype(int)

    # delete [ and ] in the column 'modbus_response' and transform it to int
    df_network['modbus_response'] = df_network['modbus_response'].str.replace('[', '')
    df_network['modbus_response'] = df_network['modbus_response'].str.replace(']', '')
    df_network['modbus_response'] = df_network['modbus_response'].fillna(0)  # Replace NaN values with 0
    df_network['modbus_response'] = df_network['modbus_response'].astype(int)

    df_network['Time'] = pd.to_datetime(df_network['Time'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
    df_network_downsampled = downsample_data(df_network, "attack_number", 0.01)
    df_network_downsampled = df_network_downsampled.rename(columns={"label": "Label", "label_n": "Label_n"})
    return df_network_downsampled



def plot_feature(df, feature, title, streamlit=False):
    fig = px.line(df, x='Time', y=feature, color='Label', title=title)
    if streamlit:
        st.plotly_chart(fig)
    else:
        fig.show()
        
def plot_feature_importances_plotly(model, columns):
    # Plot feature importances
    # if model is not KNN
    if hasattr(model, 'feature_importances_'):
        fig = go.Figure()
        fig.add_trace(go.Bar(x=columns, y=model.feature_importances_))
        fig.update_layout(title="Feature Importance", xaxis_title="Feature", yaxis_title="Feature importance")
        st.plotly_chart(fig)
    else:
        st.write("- KNN model does not have feature importances")
        
        
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

