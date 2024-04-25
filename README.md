# Power Line Fault Detection
This project involves the development of a machine learning model for detecting faults in power lines using sensor data.

## Project Overview
The goal of this project is to detect different types of faults in power lines, such as LG fault (Between Phase A and Gnd), LL fault (Between Phase A and Phase B), LLG Fault (Between Phases A,B and ground), LLL Fault(Between all three phases), and LLLG fault( Three phase symmetrical fault).

The model is trained on a dataset that contains sensor readings from power lines, including line voltages and currents. The labels in the dataset represent different types of faults.
The model used in this project is a Random Forest Classifier from the Scikit-learn library. The Random Forest model was chosen for its ability to handle high-dimensional datasets and its robustness to outliers and overfitting.

## Project details
Here are some details about the model and its training process:

### Model: Random Forest Classifier
Number of Estimators: 100 (default)
Number of Layers: N/A (Random Forests do not have layers in the same sense as neural networks)
Epochs: N/A (Random Forests are not trained iteratively like neural networks)
Model Efficiency: This would be the performance metrics of your model such as accuracy, precision, recall, F1-score, etc. on your validation/test set. You would fill in these details based on the results of your model evaluation.
### Files in this Repository
model.py: This file contains the code for creating the Random Forest model.
train.py: This file contains the code for training the model on the dataset.
validate.py: This file contains the code for validating the model on a test set.
main.py: This file contains the code for making predictions on new data.
## How to Use
Run train.py to train the model. This will create a file model.pkl which is the trained model.
Run validate.py to validate the model on a test set.
Run main.py to use the trained model to make predictions on new data.
