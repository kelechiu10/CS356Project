from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from models.base_model import BaseModel
class SVM(BaseModel):
    def train(self, data):
        X_train = data['X_train']
        y_train = data['y_train']
        clf = SVC(gamma='auto')
        clf.fit(X=X_train, y=y_train)

        print("Accuracy score on Validation set: \n")
        print(clf.score(X_train, y_train))

        return clf

    def test(self, model, data):
        X_test = data['X_test']
        y_test = data['y_test']
        predictions = model.predict(X_test)
        print(f'accuracy: {accuracy_score(y_test, predictions)}')
        cm = confusion_matrix(y_test, predictions)
        print(cm)
        print(classification_report(y_test, predictions, digits=5))