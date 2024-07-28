# Hand-written-Digit-Classification
The digits dataset consists of 8x8 pixel images of digits. the images attribute of the dataset stores 8x8 arrays of grayscale values for each image. we will use these arrays to visualize the dirst 4 images. the target attribute of the dataset stores the digit each image represents

##Explaination
A Random Forest Classifier model is created and trained on the training data.

The trained model is used to predict the digits on the testing set, and the model's performance is evaluated using a confusion matrix and classification report.

Here is a breakdown

import Libraries:

pandas: Used for data manipulation and analysis.

numpy: Used for numerical operations on arrays.

matplotlib.pyplot: Used for creating visualizations.

sklearn.datasets: Used to load the handwritten digits dataset.

sklearn.model_selection.train_test_split: Used to split the data into training and testing sets.

sklearn.ensemble.RandomForestClassifier: The machine learning model used for classification.

sklearn.metrics.confusion_matrix, classification_report: Used to evaluate the performance of the model.

Load and Visualize Data:

The handwritten digits dataset is loaded using load_digits().

The first four images are displayed using matplotlib to visualize the data.

Preprocess Data:

Reshape Images: The images are reshaped from 8x8 arrays to a flat array of 64 features using reshape((n_samples, -1)).

Scale Pixel Values: The pixel values are scaled from a range of 0-16 to 0-1 by dividing by 16. This helps improve the performance of the machine learning model.

Split Data:

The data is split into training and testing sets using train_test_split with a test size of 30%.

Train Model:

A Random Forest Classifier model is created using RandomForestClassifier().

The model is trained on the training data using fit(X_train, y_train).

Predict and Evaluate:

The trained model is used to predict the digits on the testing set using predict(X_test).

A confusion matrix is generated using confusion_matrix(y_test, y_pred) to visualize the model's performance.

A classification report is generated using classification_report(y_test, y_pred) to provide a detailed evaluation of the model's precision, recall, F1-score, and support for each digit class.
