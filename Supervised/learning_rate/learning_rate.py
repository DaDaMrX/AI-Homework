from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import pickle

if __name__ == '__main__':
    digits = datasets.load_digits()
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    labels = digits.target

    clf = MLPClassifier(hidden_layer_sizes=200)

    learning_rate_init = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005,
                          0.0006, 0.0007, 0.0008, 0.0009,
                          0.001, 0.002, 0.003, 0.004, 0.005,
                          0.006, 0.007, 0.008, 0.009,
                          0.01, 0.02, 0.03, 0.04, 0.05]
    param_grid = [{
        'solver': ['sgd', 'adam'],
        'learning_rate_init': learning_rate_init
    }]
    grid = GridSearchCV(clf, param_grid, scoring='f1_micro', cv=5, n_jobs=4,
                        pre_dispatch='2*n_jobs', verbose=3, refit=False)
    grid.fit(data, digits.target)

    with open('_learning_rate.pkl', 'wb') as f:
        pickle.dump(grid.cv_results_, f)
