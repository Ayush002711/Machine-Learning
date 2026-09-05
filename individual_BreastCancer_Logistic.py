import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

#--------------------------
#Step 1 :Load the Dataset
#---------------------------

Border="-"*40
df=pd.read_csv("breast_cancer.csv")

print("Shape of Datasets:",df.shape)

print(Border)
print("First Few Records:")
print(Border)
print(df.head())

#-------------------------------------
#Step 2 :Seperate features and labels
#-------------------------------------

X=df.drop("target",axis=1)
Y=df["target"]

print("X Shape :",X.shape)
print("Y Shape:",Y.shape)

#-----------------------------------------------
#Step 3 :Split dataset for training and testing
#-----------------------------------------------

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

#-----------------------------------------------
#Step 4 :Scale the features
#-----------------------------------------------

scalar=StandardScaler()

X_train=scalar.fit_transform(X_train)
X_test=scalar.fit_transform(X_test)

#-----------------------------------------------
#Step 5 :Create the Model
#-----------------------------------------------

model=LogisticRegression(max_iter=1000)

#-----------------------------------------------
#Step 6 :Train the Model
#-----------------------------------------------

model=model.fit(X_train,Y_train)

#--------------------------------------------
#Step 7 :Test the Model
#--------------------------------------------

Y_pred=model.predict(X_test)

#--------------------------------------------
#Step 8 :Evaluate the Model
#--------------------------------------------
print(Border)
print("Accuracy:",accuracy_score(Y_test,Y_pred))
print(Border)

print(Border)
print("Confussion Matrix:")
print(Border)
print(confusion_matrix(Y_test,Y_pred))
print(Border)


      




