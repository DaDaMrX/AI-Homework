import matplotlib.pyplot as plt
import pickle

with open('learning_rate.pkl', 'rb') as f:
    result = pickle.load(f)

params = result['params']
mean_test_score = result['mean_test_score']

learning_rate_init = []
sgd_score = []
adam_score = []
for param, score in zip(params, mean_test_score):
    if param['solver'] == 'sgd':
        learning_rate_init.append(param['learning_rate_init'])
        sgd_score.append(score)
    elif param['solver'] == 'adam':
        adam_score.append(score)

x = list(range(1, len(learning_rate_init) + 1))
is_tick = [0, 4, 9, 13, 18, 22]
xticks = [''] * len(learning_rate_init)
for i in is_tick:
    xticks[i] = str(learning_rate_init[i])

plt.subplots(figsize=(10, 5))
plt.xticks(x, xticks)
plt.plot(x, sgd_score, label='SGD', color='b')
plt.plot(x, adam_score, label='Adam', color='r')

plt.xlabel('Initial learning rate')
plt.ylabel('Micro-f1 score')
plt.title('The micro-f1 score of different initial learning rate')
plt.legend()
plt.savefig('learning_rate')
plt.show()
