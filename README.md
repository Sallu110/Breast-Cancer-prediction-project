
# Breast Cancer Prediction Project
This project aims to predict breast cancer in patients based on various features using machine learning models. Specifically, the Random Forest Classifier is utilized along with Principal Component Analysis (PCA) for feature selection. The project includes data preprocessing, model training, and evaluation.

## Steps of project
1. Dataset
2. Dependencies
3. Implementation
4. Loading the Dataset
5. Data Preprocessing
6. Feature selection
7. Model Evaluation
8. Conclusion

### dataset
The dataset used in this project is the Breast Cancer Wisconsin (Diagnostic) dataset, which can be loaded directly from the sklearn.datasets module.

### Dependencies
The project requires the following Python libraries:
pandas
numpy
scikit-learn

### implementation
Loading the Dataset
The dataset is loaded using the load_breast_cancer function from sklearn.datasets. The features and target variables are then extracted into separate DataFrames.
``` python
from sklearn.datasets import load_breast_cancer
import pandas as pd

lbc = load_breast_cancer()
X = pd.DataFrame(lbc['data'], columns=lbc['feature_names'])
Y = pd.DataFrame(lbc['target'], columns=['type'])

# Data Preprocessing
# The dataset is split into training and testing sets using the train_test_split function. Data scaling is performed using StandardScaler.

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1234, stratify=Y)
scaler = StandardScaler()
x_scaled = scaler.fit_transform(X)

# Random Forest Classifier
# The Random Forest Classifier is trained on the training data, and predictions are made on the test data.

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(random_state=1234)
rfc.fit(X_train, Y_train)
R_predict = rfc.predict(X_test)
```
### feature selection
``` python
# Principal Component Analysis (PCA) for autofeature selection
# PCA is implemented to reduce the dimensionality of the dataset, and the Random Forest Classifier is retrained on the transformed data.

# from sklearn.decomposition import PCA

pca = PCA(n_components=5)
x_pca = pca.fit_transform(x_scaled)
X_train_pca, X_test_pca, Y_train, Y_test = train_test_split(x_pca, Y, test_size=0.3, random_state=1234, stratify=Y)

rfc2 = RandomForestClassifier(random_state=1234)
rfc2.fit(X_train_pca, Y_train)
R_predict2 = rfc2.predict(X_test_pca)
```
### Model Evaluation
``` python
The models are evaluated using confusion matrix and accuracy score.
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, R_predict)
score = rfc.score(X_test, Y_test)
```
![Screenshot 2024-07-18 165232](https://github.com/user-attachments/assets/ebe7a60a-5243-4776-9ae5-64d82be09cce)

``` python
cm2 = confusion_matrix(Y_test, R_predict2)
score2 = rfc2.score(X_test_pca, Y_test)
```
![Screenshot 2024-07-18 165256](https://github.com/user-attachments/assets/957d60ad-93ae-4ae1-b333-b35db4b8ee91)

![Screenshot 2024-07-18 170200](https://github.com/user-attachments/assets/0e79ff6b-2acb-45df-83e2-ee5ce1bede5d)

### Conclusion
This project demonstrates the use of Random Forest Classifier and PCA for breast cancer prediction. The results indicate the effectiveness of PCA in improving model performance. Future work can include testing other machine learning models and feature selection techniques to further enhance prediction accuracy.






