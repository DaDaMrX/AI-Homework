from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import pickle

if __name__ == '__main__':
    digits = datasets.load_digits()
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    labels = digits.target

    clf = MLPClassifier(learning_rate_init=0.001)

    hidden_layer_sizes = list(range(100, 501, 20))
    param_grid = [{
        'solver': ['sgd', 'adam'],
        'hidden_layer_sizes': hidden_layer_sizes,
    }]
    grid = GridSearchCV(clf, param_grid, scoring='f1_micro', cv=5, n_jobs=4,
                        pre_dispatch='2*n_jobs', verbose=3, refit=False)
    grid.fit(data, digits.target)

    with open('hidden_layer.pkl', 'wb') as f:
        pickle.dump(grid.cv_results_, f)
