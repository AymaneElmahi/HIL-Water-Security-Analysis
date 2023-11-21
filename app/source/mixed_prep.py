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
from .physical_prep import *

def mix_prep(root):
    df_physical_raw, df_physical = physical_prep(root)
    del df_physical_raw
    df_network_downsampled = pd.read_csv(os.path.join(root, "Network datatset/csv/network_downsampled.csv"), encoding='utf-8-sig', sep=',')
    df_network_downsampled['Time'] = pd.to_datetime(df_network_downsampled['Time'], format='%Y-%m-%d %H:%M:%S.%f')    # Round the timestamps with milliseconds to the nearest second in df_network
    df_network_downsampled['Time'] = df_network_downsampled['Time'].dt.round('1s')

    # Merge the two dataframes based on the 'Time' and 'attack type number' and Label_n and Label columns and drop the 'attack' column
    merged_df = pd.merge(df_network_downsampled, df_physical, on=['Time', 'attack_type_number', 'Label_n', 'Label'], how='inner')

    # delete columns with only one value
    for col in merged_df.columns:
        if len(merged_df[col].unique()) == 1:
            merged_df = merged_df.drop(col, axis=1)
            
    return merged_df


def plot_metrics(y_test, y_pred, X_test, model):
    # Classification report
    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))

    # Confusion Matrix Plot
    conf_mat = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))  # Adjust the figure size
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix ')
    # Save the plot to a buffer
    buffer_conf_mat = io.BytesIO()
    plt.savefig(buffer_conf_mat, format='png')
    # Display the confusion matrix plot with a specified width
    st.subheader("Confusion Matrix")
    st.image(buffer_conf_mat.getvalue(), width=600)  # Adjust the width
    # check if multiple classes
    if len(np.unique(y_test)) > 2:
        st.write("ROC Curve and AUC are not available for multiple classes")
        return
    
    st.write("ROC Curve and AUC are not available for multiple classes")
    # ROC Curve and AUC
    y_probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    # ROC Curve Plot
    plt.figure(figsize=(8, 6))  # Adjust the figure size
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    # Save the plot to a buffer
    buffer_roc_curve = io.BytesIO()
    plt.savefig(buffer_roc_curve, format='png')
    # Display the ROC curve plot with a specified width
    st.subheader("ROC Curve")
    st.image(buffer_roc_curve.getvalue(), width=600)  # Adjust the width

def plot_feature_importances(model, columns):
    # Plot feature importances
    plt.figure(figsize=(8, 6))  # Adjust the figure size
    n_features = len(columns)
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.title("Feature Importance")
    # Save the plot to a buffer
    buffer_feature_importances = io.BytesIO()
    plt.savefig(buffer_feature_importances, format='png')
    # Display the feature importances plot with a specified width
    st.subheader("Feature Importance")
    st.image(buffer_feature_importances.getvalue(), width=600) # Adjust the width
    
# do it using plotly
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
        