from sklearn import datasets
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt

digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
labels = digits.target
split_point = int(n_samples * 0.8)

clf = svm.SVC(gamma=0.001)
clf.fit(data[:split_point], labels[:split_point])

predicted = clf.predict(data[split_point:])
expected = labels[split_point:]

print('Classification report:')
print(metrics.classification_report(expected, predicted))
print()
print('Confusion matrix:')
print(metrics.confusion_matrix(expected, predicted))

images_and_labels = list(zip(digits.images, digits.target))
for index, (image, label) in enumerate(images_and_labels[:4]):
    plt.subplot(2, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)

images_and_predictions = list(zip(digits.images[split_point:], predicted))
for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(2, 4, index + 5)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)

plt.suptitle('Result of Support vector machine')
plt.savefig('SVM.png')
plt.show()
