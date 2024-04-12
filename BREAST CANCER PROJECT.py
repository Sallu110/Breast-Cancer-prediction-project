

# for the breast cancer prediction and campare results 
# implement principal component analysis
 
from sklearn.datasets import load_breast_cancer
import pandas as pd 

lbc = load_breast_cancer()

X =pd.DataFrame(lbc['data'], columns = lbc['feature_names'])
Y = pd.DataFrame(lbc['target'],columns = ['type'])


from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.3,random_state = 1234, stratify=Y)

# Random forest classifier 

from sklearn.ensemble import RandomForestClassifier

# TRAIN THE MODEL 

rfc = RandomForestClassifier(random_state = 1234)
rfc.fit(X_train,Y_train)
R_predict = rfc.predict(X_test)

# Evaluate the model 
from sklearn.metrics import confusion_matrix 

cm    = confusion_matrix(Y_test,R_predict)
score = rfc.score(X_test,Y_test)    

# Center the data 

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

x_scaled = scaler.fit_transform(X)

x_scaled[:,0].mean()


# implement pca 


from sklearn.decomposition import PCA

pca = PCA(n_components=5)
x_pca = pca.fit_transform(x_scaled)



X_train,X_test,Y_train,Y_test = train_test_split(x_pca,Y,test_size = 0.3,random_state = 1234, stratify=Y)

# Random forest classifier 
# TRAIN THE MODEL 

rfc2 = RandomForestClassifier(random_state = 1234)
rfc2.fit(X_train,Y_train)
R_predict2 = rfc2.predict(X_test)

# Evaluate the model 
from sklearn.metrics import confusion_matrix 

cm2    = confusion_matrix(Y_test,R_predict2)
score2 = rfc2.score(X_test,Y_test)    

PROJECT END 

