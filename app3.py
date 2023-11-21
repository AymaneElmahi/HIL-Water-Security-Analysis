import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
# roc 
from sklearn.metrics import roc_curve, auc
import io
import os


# Load the DataFrame
df = pd.read_csv('dataset/Physical dataset/phy_all.csv')  # Replace 'your_data.csv' with the actual file path
df_physical = df.copy()

# Section 1: Data Overview
st.title('Data Overview')
st.write("## Descriptive Statistics")
st.write(df_physical.describe())

# Section 2: Histograms
st.title('Histograms')
# Create histograms and display using st.pyplot()
df_physical.hist(figsize=(20, 20), bins=50)
st.pyplot()

# Section 3: Deleted Columns
st.title('Deleted Columns')
st.write("Let's delete columns with only one unique value.")
# Delete columns with only one unique value
for col in df_physical.columns:
    if len(df_physical[col].unique()) == 1:
        df_physical.drop(col, axis=1, inplace=True)

# Display updated information after deleting columns
st.write("## Updated Information After Deleting Columns")
# Redirect printed output to a string variable
buffer = io.StringIO()
df_physical.info(buf=buffer)
info_str = buffer.getvalue()
# Display the information string
st.text(info_str)

# Section 4: Time Series Plots
st.title('Time Series Plots')
# Function to plot tank level against flow rate
attack_numbers = [0, 1, 2, 3, 4]
def plot_tank_column_over_time(column_name, attack_numbers, df_copy):
    # Check if the plots directory exists, create it if not
    plots_directory = "plots"
    if not os.path.exists(plots_directory):
        os.makedirs(plots_directory)
    # Construct the filename
    filename = f"{column_name}_overtime_attack_number.png"
    plot_path = os.path.join(plots_directory, filename)
    if os.path.exists(plot_path):
        # If the plot image already exists, load and display it
        st.image(plot_path)
    else:
        # Set up a figure with subplots
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
        fig.suptitle(f'{column_name} over time with different attack numbers')
        for i, ax in enumerate(axes.flat):
            if i < len(attack_numbers):
                attack_num = attack_numbers[i]
                attack_filter = df_copy['attack number'] == attack_num
                ax.plot(df_copy[attack_filter]['Time'], df_copy[attack_filter][column_name], label=f'Attack {attack_num}')
                ax.set_title(f'Attack {attack_num}')
                ax.set_xlabel('Time')
                ax.set_ylabel(column_name)
                ax.legend()
        # Adjust layout for better spacing
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        # Save the plot to the plots directory
        plt.savefig(plot_path)
        # Display the plot
        st.image(plot_path)
        # Close the plot to release resources
        plt.close()

df_copy = df_physical.copy()
# Plot Tank_1 as an example
plot_tank_column_over_time('Tank_2', [0, 1, 2, 3, 4], df_copy)
# plot Flow_sensor_1 as an example
plot_tank_column_over_time('Flow_sensor_1', [0, 1, 2, 3, 4], df_copy)
# plot Valv_10 as an example
plot_tank_column_over_time('Valv_10', [0, 1, 2, 3, 4], df_copy)

# Section 5: Heatmap of Correlation Matrix
st.title('Correlation Matrix Heatmap')
numeric_columns = df_physical.select_dtypes(include=[float, int]).columns
correlation_matrix = df_physical[numeric_columns].corr()
plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
st.pyplot()
st.write("### Strongest Correlations:")
# Your correlation matrix content...

# Section 6: Model Training and Evaluation
st.title('Model Training and Evaluation')
# Prepare data for modeling
X = df_physical.drop(['Label', 'Label_n', 'Time', 'attack number'], axis=1)
y = df_physical['Label_n']
# User chooses between Decision Tree and Random Forest
model_choice = st.radio("Choose a Model:", ("Decision Tree", "Random Forest"))
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def plot_metrics(y_test, y_pred):
    # classification report
    st.header("Classification Report")
    st.text(classification_report(y_test, y_pred))
    # Confusion Matrix Plot
    conf_mat = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    # Save the plot to a buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    # Display the confusion matrix plot with a specified width
    st.image(buffer.getvalue(), width=900)
    # ROC Curve and AUC
    y_probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    # ROC Curve Plot
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    # Display the ROC curve plot
    st.pyplot()

# Train the selected model
if model_choice == "Decision Tree":
    model = DecisionTreeClassifier()
elif model_choice == "Random Forest":
    model = RandomForestClassifier()
else:
    st.error("Invalid model choice. Please choose either 'Decision Tree' or 'Random Forest'.")

# Check if the model is selected before proceeding
if 'model' in locals():
    model.fit(X_train, y_train)
    # Evaluate the model
    y_pred = model.predict(X_test)
    plot_metrics(y_test, y_pred)
