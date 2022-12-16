from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from models.base_model import BaseModel
class XGBoost(BaseModel):
    def __init__(self, gpu=False) -> None:
        super().__init__()
        self.gpu = gpu
    def train(self, data):
        X_train = data['X_train']
        y_train = data['y_train']

        if self.gpu:
            model = XGBClassifier(tree_method='gpu_hist')
        else:
            model = XGBClassifier()
        params = {
            'min_child_weight': [1, 5, 10],
            'gamma': [0.5, 1, 1.5, 2, 5],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'max_depth': [3, 4, 5]
            }
        clf = RandomizedSearchCV(
            estimator=model,
            param_distributions= params,
            cv=5,
            verbose=3,
            n_iter=200,
            random_state=42,
            n_jobs=-1,  # Use all available CPU cores
            return_train_score=True
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