import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
cancer=pd.read_csv("cancer.csv")
y = cancer['diagnosis']
X = cancer.drop(['id','diagnosis','Unnamed: 32'],axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7, random_state=2529)

model = LogisticRegression(max_iter=5000)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
y_pred
confusion_matrix(y_test,y_pred)
accuracy_score(y_test,y_pred)
with open("breast_cancer_model.pkl", "wb") as f:
    pickle.dump(model, f)

