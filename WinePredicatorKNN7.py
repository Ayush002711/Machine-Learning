import pandas as pd
import matplotlib.pylab as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.preprocessing import StandardScaler

def MarvellousClassifier(Datapath):
    Border="-"*40

    print(Border)
    print("Step1 : Load the Dataset from CSV file")
    print(Border)

    df=pd.read_csv(Datapath)

    print(Border)
    print("Some entries from Dataset")
    print(df.head())
    print(Border)

    print(Border)
    print("Step2: Clean the Dataset")
    print(Border)

    df.dropna(inplace=True)
    print("Shape of Datset:",df.shape)
    print("Total records:",df.shape[0])
    print("Total Columns:",df.shape[1])

    print(Border)

    print(Border)
    print("Step3:Seperate Independent and Dependent variabales ")
    print(Border)

    X=df.drop(columns=["Class"])
    Y=df["Class"]

    print("Shape of X:",X.shape)
    print("Shape of Y:",Y.shape)

    print("Input Columns :",X.columns.tolist())
    print("Output Colimns: Class")

    print(Border)

    print(Border)
    print("Step4: Split the Dataset for training and testing")
    print(Border)

    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42,stratify=Y)
    print("Details of training and testing data")

    print("Shape of X_train:",X_train.shape)
    print("Shape of X_test:",X_test.shape)
    print("Shape of Y_train:",Y_train.shape)
    print("Shape of Y_test:",Y_test.shape)

    print(Border)

    print(Border)
    print("Step5: Feature Scaling")
    print(Border)

    sobj=StandardScaler()
    X_train_scaled=sobj.fit_transform(X_train)
    X_test_scaled=sobj.fit_transform(X_test)

    print("Feature Scaling Done")
    print(Border)

    print(Border)
    print("Step6:Hyperparameter Tuning")
    print(Border)

    accuracy_score=[]
    K_values=range(1,21)

    for k in K_values:
        model=KNeighborsClassifier(n_neighbors=k)
        model = model.fit(X_test_scaled,Y_train)
        Y_pred=model.predict(X_test_scaled)
        Accuracy=accuracy_score(Y_test,Y_pred)
        accuracy_score.append(Accuracy)

    print("Accuracy Report :")
    for No in accuracy_score:
        print(No)

    print(Border)

    print(Border)
    print("Step7:Graphical Represntation")
    print(Border)

    plt.figure(figsize=(8,5))
    plt.plot(K_values,accuracy_score,marker="o")
    plt.title("k values Vs Accuracy ")
    plt.xlabel("value of k")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.xticks(list(K_values))
    plt.show()

    
def main():
    MarvellousClassifier("WinePredictor.csv")

if __name__=="__main__":
    main()

