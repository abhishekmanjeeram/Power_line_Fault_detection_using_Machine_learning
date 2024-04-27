import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# Load your data 
data = pd.read_csv('datasets/train.csv')
X = data[['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']].values
y = data[['G', 'C', 'B', 'A']].values

# Map binary labels to multi-class labels
class_mapping = {
    (0, 0, 0, 0): 'No Fault',
    (1, 0, 0, 1): 'LG fault',
    (0, 0, 1, 1): 'LL fault',
    (1, 0, 1, 1): 'LLG Fault',
    (0, 1, 1, 1): 'LLL Fault',
    (1, 1, 1, 1): 'LLLG fault'
}

# Initialize an empty list to store the multi-class labels
y_multi_class = []

# Iterate through each label and map it to the corresponding multi-class label
for label in y:
    try:
        y_multi_class.append(class_mapping[tuple(label)])
    except KeyError:
        # Handle the exception (unknown label) by assigning it to 'Unknown'
        y_multi_class.append('Unknown')

# Split your data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y_multi_class, test_size=0.2, random_state=42)

# Initialize the classifier
clf = RandomForestClassifier()

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = clf.predict(X_val)

# Generate the confusion matrix
cm = confusion_matrix(y_val, y_pred)

# Define class labels
class_labels = list(class_mapping.values())

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=class_labels, yticklabels=class_labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
