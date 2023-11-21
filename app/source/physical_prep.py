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



def physical_prep(root):
    df_physical_normal = pd.read_csv(os.path.join(root, "Physical dataset/phy_norm.csv"), encoding='utf-16-le', sep='\t')
    df_physical_att_1 = pd.read_csv(os.path.join(root, "Physical dataset/phy_att_1.csv"), encoding='utf-16-le', sep='\t')
    df_physical_att_2 = pd.read_csv(os.path.join(root, "Physical dataset/phy_att_2.csv"), encoding='utf-16-le', sep='\t')
    df_physical_att_3 = pd.read_csv(os.path.join(root, "Physical dataset/phy_att_3.csv"), encoding='utf-16-le', sep='\t')
    df_physical_att_4 = pd.read_csv(os.path.join(root, "Physical dataset/phy_att_4.csv"), encoding='utf-8-sig', sep=',')

    df_physical_att_1.columns = df_physical_normal.columns
    df_physical_att_2.columns = df_physical_normal.columns
    df_physical_att_3.columns = df_physical_normal.columns
    df_physical_att_4.columns = df_physical_normal.columns
    
    df_physical_normal['attack_number'] = 0
    df_physical_att_1['attack_number'] = 1
    df_physical_att_2['attack_number'] = 2
    df_physical_att_3['attack_number'] = 3
    df_physical_att_4['attack_number'] = 4
    
    df_physical_raw = pd.concat([df_physical_normal, df_physical_att_1, df_physical_att_2, df_physical_att_3, df_physical_att_4])
    
    for col in df_physical_raw.columns:
        if df_physical_raw[col].dtype == 'bool':
            df_physical_raw[col] = df_physical_raw[col].astype(int)
        
    df_physical= df_physical_raw.copy()
    
            
    for col in df_physical.columns:
        if len(df_physical[col].unique()) == 1:
            df_physical.drop(col, axis=1, inplace=True)
            
    df_physical['Label'] = df_physical['Label'].replace('nomal', 'normal')
    df_physical = df_physical[df_physical['Label'] != 'scan']
    df_physical['attack_type_number'] = df_physical['Label'].map({'normal': 0, 'MITM': 1, 'physical fault': 2, 'DoS': 3})
    df_physical['Time'] = pd.to_datetime(df_physical['Time'], format='%d/%m/%Y %H:%M:%S')    
    return df_physical_raw, df_physical






def plot_feature(df, feature, title, streamlit=False):
    fig = px.line(df, x='Time', y=feature, color='Label', title=title)
    if streamlit:
        st.plotly_chart(fig)
    else:
        fig.show()
        
def plot_feature_feature(feature_1, feature_2, df):
    fig = go.Figure()

    # plot feature_1 in blue and feature_2 in green
    fig.add_trace(go.Scatter(y=df[feature_1], mode='lines', name=feature_1, line=dict(color='blue')))
    fig.add_trace(go.Scatter(y=df[feature_2], mode='lines', name=feature_2, line=dict(color='green')))

    fig.update_layout(title=f'{feature_1} vs. {feature_2}', xaxis_title=feature_1, yaxis_title=feature_2)

    st.plotly_chart(fig)
    
    
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
