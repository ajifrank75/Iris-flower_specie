import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Importing the dataset
df = pd.read_csv('Iris.csv')
print(df.head())

# check for null values in the dataset
df.isnull().any()

# Select the columns of independent and independent variables
x = df.iloc[:, 1:5].values 
y = df.iloc[:, [5]].values 

# Encode the target species to integers for easy calculation all through
def prepare_targets(y_enc):
    le = LabelEncoder()
    le.fit(y_enc)
    y_enc = le.transform(y)
    return y_enc

# The encoded target
y_encoded = prepare_targets(y)
print(y_encoded)

# Plot the graph for visualization
sns.pairplot(df, hue='Species')
plt.savefig('output.png',dpi=300)

# splitting the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=.3,random_state=0)

# fitting Decision Tree to trainning set
classifier = DecisionTreeClassifier()
classifier.fit(x_train, y_train)

#predicting test set resultss
predictions = classifier.predict(x_test)

#confusion matrix
cm = confusion_matrix(y_test,predictions)
print(cm)

# check for accuracy
print("Accuracy score: ", accuracy_score(y_test, predictions)* 100, "%")
