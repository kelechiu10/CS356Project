from sklearn.neighbors import KNeighborsClassifier
from .base_model import BaseModel
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class KNN(BaseModel):
    def train(self, data):
        X_train = data['X_train']
        y_train = data['y_train']
        model = KNeighborsClassifier(n_neighbors=2)
        hyperparameters = {
            'leaf_size': [20, 30, 40]
        }
        clf = GridSearchCV(
            estimator=model,
            param_grid=hyperparameters,
            cv=5,
            verbose=10,
            n_jobs=-1  # Use all available CPU cores
        )
        clf.fit(X=X_train, y=y_train)

        print("Accuracy score on Validation set: \n")
        print(clf.best_score_ )
        print("---------------")
        print("Best performing hyperparameters on Validation set: ")
        print(clf.best_params_)
        print("---------------")
        print(clf.best_estimator_)

        return clf.best_estimator_

    def test(self, model, data):
        X_test = data['X_test']
        y_test = data['y_test']
        predictions = model.predict(X_test)
        print(f'accuracy: {accuracy_score(y_test, predictions)}')
        cm = confusion_matrix(y_test, predictions)
        print(cm)
        print(classification_report(y_test, predictions, digits=5))