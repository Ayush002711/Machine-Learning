# Data Visualization & Analytics(CaseStudyStep9)

##  Project Overview

This project implements a **Machine Learning classification model** to predict the species of an Iris flower using its sepal and petal measurements.

The **Decision Tree Classifier** from Scikit-learn is used to train the model and classify Iris flowers into different species.

##  Dataset

The project uses the `iris.csv` dataset.

The dataset contains four input features:

* Sepal Length
* Sepal Width
* Petal Length
* Petal Width

### Target Variable

* Species

The model predicts the Iris flower species based on these four measurements.

##  Machine Learning Algorithm

**Decision Tree Classifier**

```python
DecisionTreeClassifier(max_depth=5)
```

The Decision Tree learns patterns from the training data and uses a series of decision rules to classify new observations.

##  Project Workflow

```text
Dataset
   ↓
Data Loading
   ↓
Exploratory Data Analysis
   ↓
Feature & Target Selection
   ↓
Data Visualization
   ↓
Train-Test Split
   ↓
Decision Tree Model
   ↓
Model Training
   ↓
Prediction
   ↓
Model Evaluation
```

##  Exploratory Data Analysis

The program performs:

* Dataset shape analysis
* Column name identification
* Missing-value checking
* Species/class distribution
* Statistical summary using `describe()`

##  Data Visualization

A scatter plot is created using:

* X-axis → Petal Length
* Y-axis → Petal Width

Different Iris species are displayed separately to understand their distribution.

##  Model Training

The dataset is divided into:

* **50% Training Data**
* **50% Testing Data**

The model is trained using:

```python
model.fit(X_train, Y_train)
```

##  Model Prediction

After training, the model predicts the species of the test dataset:

```python
Y_pred = model.predict(X_test)
```

##  Model Evaluation

The model performance is evaluated using:

### Accuracy

Measures the percentage of correctly classified samples.

### Confusion Matrix

Shows the number of correct and incorrect predictions for each class.

### Classification Report

Provides:

* Precision
* Recall
* F1-score
* Support

##  Technologies Used

* Python
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn

##  Python Libraries

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
```

##  How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
```

### 2. Navigate to the project folder

```bash
cd Iris-Decision-Tree
```

### 3. Install required libraries

```bash
pip install pandas matplotlib seaborn scikit-learn
```

### 4. Run the Python program

```bash
python iris_decision_tree.py
```

##  Project Structure

```text
Iris-Decision-Tree/
│
├── iris.csv
├── iris_decision_tree.py
├── README.md
└── images/
```

##  Learning Objectives

Through this project, I practiced:

* Loading datasets using Pandas
* Exploratory Data Analysis
* Feature and target selection
* Data visualization
* Train-test splitting
* Decision Tree classification
* Model prediction
* Accuracy calculation
* Confusion matrix
* Classification report

##  Author

**Ayush Jadhav**

---

⭐ If you find this project useful, consider giving the repository a star!
