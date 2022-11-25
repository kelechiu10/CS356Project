from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import class_weight
import numpy as np
from sklearn.model_selection import GridSearchCV
from .base_model import BaseModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class RandomForest(BaseModel):
    """Random Forest"""
    def train(self, data):
        X_train = data['X_train']
        y_train = data['y_train']

        # Calculating class weights for balanced class weighted classifier training
        class_weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )

        # Must be in dict format for scikitlearn
        class_weights = {
            0: class_weights[0],
            1: class_weights[1]
        }

        model = RandomForestClassifier(
            n_estimators=100,
            criterion='gini',
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features='auto',
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            bootstrap=True,
            oob_score=False,
            n_jobs=None,
            random_state=None,
            verbose=1,
            warm_start=False,
            class_weight=class_weights,
            ccp_alpha=0.0,
            max_samples=None
        )

        hyperparameters = {
            'n_estimators': [50, 75, 100, 125, 150]
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
