from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from models.base_model import BaseModel
class DecisionTree(BaseModel):
    def train(self, data):
        X_train = data['X_train']
        y_train = data['y_train']
        model = DecisionTreeClassifier(
            criterion='gini',
            splitter='best',
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features=None,
            random_state=None,
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            class_weight=None,
            ccp_alpha=0.0
        )

        hyperparameters = {
            'max_depth': [i for i in range(1, 20)]
        }
        clf = GridSearchCV(
            estimator=model,
            param_grid=hyperparameters,
            cv=5,
            verbose=1,
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
