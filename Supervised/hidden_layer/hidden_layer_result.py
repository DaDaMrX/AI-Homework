import matplotlib.pyplot as plt
import pickle

with open('hidden_layer.pkl', 'rb') as f:
    result = pickle.load(f)

params = result['params']
mean_test_score = result['mean_test_score']

hidden_layer_sizes = []
sgd_score = []
adam_score = []
for param, score in zip(params, mean_test_score):
    if param['solver'] == 'sgd':
        hidden_layer_sizes.append(param['hidden_layer_sizes'])
        sgd_score.append(score)
    elif param['solver'] == 'adam':
        adam_score.append(score)

plt.subplots(figsize=(10, 5))
plt.plot(hidden_layer_sizes, sgd_score, label='SGD', color='b')
plt.plot(hidden_layer_sizes, adam_score, label='Adam', color='r')

plt.xlabel('Hidden layer sizes')
plt.ylabel('Micro-f1 score')
plt.title('The micro-f1 score of different hidden layer sizes')
plt.legend()
plt.savefig('hidden_layer')
plt.show()
