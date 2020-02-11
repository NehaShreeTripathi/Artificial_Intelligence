# Import the important library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('Iris.csv').drop("Id", axis=1)
df.head()
sns.pairplot(df, hue="Species", size=2, diag_kind="kde");
#Feature Extraction and Test-Train-Split
df = df.replace(['Iris-versicolor','Iris-virginica','Iris-setosa'],[0, 1, 2])
X = df.drop("Species",axis=1)
y  = df["Species"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# Import the models from sklearn
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

objects = ('Multi-NB', 'DTs', 'AdaBoost', 'KNN', 'RF')
# Initialize the three models
A = MultinomialNB(alpha=1.0,fit_prior=True)
B = DecisionTreeClassifier(random_state=42)
C = AdaBoostClassifier(n_estimators=100)
D = KNeighborsClassifier(n_neighbors=3)
E = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)


# function to train classifier
def train_classifier(clf, X_train, y_train):
    clf.fit(X_train, y_train)

# function to predict features
def predict_labels(clf, features):
    return (clf.predict(features))


# loop to call function for each model
clf = [A,B,C,D,E]
pred_val = [0,0,0,0,0]

for a in range(0,5):
    train_classifier(clf[a], X_train, y_train)
    y_pred = predict_labels(clf[a],X_test)
    pred_val[a] = f1_score(y_test, y_pred,  average='weighted')
    print (pred_val[a])
    # ploating data for F1 Score
    y_pos = np.arange(len(objects))
    y_val = [x for x in pred_val]
    plt.bar(y_pos, y_val, align='center', alpha=0.7)
    plt.xticks(y_pos, objects)
    plt.ylabel('Accuracy Score')
    plt.title('Accuracy of Models')
    plt.show()