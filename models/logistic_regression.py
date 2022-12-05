import sklearn.linear_model
from .base_model import BaseModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class LogisticRegression(BaseModel):
    """Logistic Regression"""
    def train(self, data):
        X_train = data['X_train']
        y_train = data['y_train']

        lr = sklearn.linear_model.LogisticRegression()
        lr.fit(X=X_train, y=y_train)

        print("Accuracy score on Validation set: \n")
        print(lr.score(X_train, y_train))

        return lr

    def test(self, model, data):

        X_test = data['X_test']
        y_test = data['y_test']
        predictions = model.predict(X_test)
        print(f'accuracy: {accuracy_score(y_test, predictions)}')
        cm = confusion_matrix(y_test, predictions)
        print(cm)
        print(classification_report(y_test, predictions, digits=5))
