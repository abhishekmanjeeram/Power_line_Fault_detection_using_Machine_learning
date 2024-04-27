import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load your data
data = pd.read_csv('datasets/train.csv')
X = data[['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']].values
y = data[['G', 'C', 'B', 'A']].values

# Split your data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the classifier
clf = RandomForestClassifier()

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = clf.predict(X_val)

# Generate the classification report
cr = classification_report(y_val, y_pred)
print("Classification Report:")
print(cr)

# Parse the classification report
lines = cr.split('\n')
classes = []
precision = []
recall = []
f1_score = []

for line in lines[2:-5]:
    t = line.split()
    if len(t) > 0:
        classes.append(t[0])
        precision.append(float(t[1]))
        recall.append(float(t[2]))
        f1_score.append(float(t[3]))

# Create bar plots
x = np.arange(len(classes))
width = 0.2

plt.figure(figsize=(10, 6))

plt.bar(x - width, precision, width, label='Precision')
plt.bar(x, recall, width, label='Recall')
plt.bar(x + width, f1_score, width, label='F1-Score')

plt.xlabel('Class')
plt.ylabel('Score')
plt.title('Performance Metrics')
plt.xticks(x, classes)
plt.legend()
plt.tight_layout()

plt.show()
