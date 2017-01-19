import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer


# create random training data again
Nclass = 500
D = 2 # dimensionality of input
M = 3 # hidden layer size
K = 3 # number of classes

X1 = np.random.randn(Nclass, D) + np.array([0, -2])
X2 = np.random.randn(Nclass, D) + np.array([2, 2])
X3 = np.random.randn(Nclass, D) + np.array([-2, 2])
X = np.vstack([X1, X2, X3]).astype(np.float32)

Y = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)
N = len(Y)


# multilabeling
lb = LabelBinarizer()
y = lb.fit_transform(Y)

# let's see what it looks like
plt.scatter(X[:,0], X[:,1], c=Y, s=100, alpha=0.5)
plt.show()

# modeling
model = Sequential()
model.add(Dense(M, input_shape=(D,)))
model.add(Activation("relu"))
model.add(Dense(K))
model.add(Activation("softmax"))
print(model.summary())

# learning
batch_size = N
nb_epoch = 1000
model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
model.fit(X, y, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1)
