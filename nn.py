from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import numpy

# seed used to get reproducable results each time
seed = numpy.random.seed(7)

# loading spam email dataset
dataset = numpy.loadtxt("dataset.csv", delimiter=",")

# split into input (X) and output (Y) variables, splitting csv data
X = dataset[:,0:57]
Y = dataset[:,57]

# define 5-fold cross validation test, specifying a random state to get repeatable results
kfold = StratifiedKFold(n_splits=5, shuffle=False, random_state=seed)
emails = []

# split X, Y into a train and test set
for train, test in kfold.split(X, Y):
    
    # create neural network model, adding layers sequentially and specifying the activation function
    neuralNetwork = Sequential()
    neuralNetwork.add(Dense(57, input_dim=57, activation='relu')) # input layer requires the number of inputs be specified
    neuralNetwork.add(Dropout(.2)) # randomly drop 20% of inputs
    neuralNetwork.add(Dense(29, activation='relu'))
    neuralNetwork.add(Dense(1,  activation='sigmoid')) # sigmoid instead of relu for final probability between 0 and 1

    # compile the model, adam gradient descent (optimized)   
    neuralNetwork.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

    # call the function to fit to the data (training the network)
    neuralNetwork.fit(X[train], Y[train], epochs = 300, batch_size=60, verbose=1)

    # evaluate the accuracy of the model
    scores = neuralNetwork.evaluate(X, Y)
    print("\n%s: %.2f%%" % ('Spam Detection Accuracy', scores[1]*100))
    emails.append(scores[1] * 100)

# print out average and standard deviation of model performance to provide a robust estimate of the model's accuracy.
print("\nOverall accuracy is %.2f%%" % numpy.mean(emails))
