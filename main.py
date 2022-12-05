import utils
from omegaconf import DictConfig
import hydra
import pandas as pd
from models import decision_tree, random_forest, knn, svm, ada, logistic_regression

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg : DictConfig) -> None:
    model_func = None
    if cfg.model == 'decision-tree':
        print('running decision tree...')
        model_func = decision_tree.DecisionTree()
    elif cfg.model == 'random-forest':
        print('running random forest...')
        model_func = random_forest.RandomForest()
    elif cfg.model == 'knn':
        print('running knn...')
        model_func = knn.KNN()
    elif cfg.model == 'svm':
        print('running SVM..')
        model_func = svm.SVM()
    elif cfg.model == 'linear_svm':
        print('running Linear SVM..')
        model_func = svm.LinearSVM()
    elif cfg.model == 'ada':
        print('running ada...')
        model_func = ada.Ada()
    elif cfg.model == 'logistic-regression':
        print('running logistic regression...')
        model_func = logistic_regression.LogisticRegression()
    else:
        raise NotImplementedError(f'{cfg.model} not added')

    df_dataset = pd.read_csv(cfg.filename)
    data = utils.process_data(df_dataset, split=(not cfg.test))
    print('data has been read in.')
    if(cfg.test):
        print('testing...')
        model = utils.load(cfg.saved_model)
        # assume whole dataset is used just for test
        model_func.test(model, data)

    else:
        print('training...')
        model = model_func.train(data)
        print('starting test')
        model_func.test(model, data)
        print(f'Saved model to trained_models/{cfg.saved_model}')
        utils.save(model, f'trained_models/{cfg.saved_model}')

if __name__ == '__main__':
    main()
