from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.utils import to_categorical

def processData(X, Y):
    # Train test split con 20% en test
    trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Train val split con 10% en validation
    trainX, valX, trainY, valY = train_test_split(trainX, trainY, test_size=0.1, random_state=42) # test_size=0.2, repeat. Try K-fold

    # For vgg19, call tf.keras.applications.vgg19.preprocess_input on your inputs before passing them to the model.
    trainX = preprocess_input(trainX)
    testX = preprocess_input(testX)
    valX = preprocess_input(valX)

    #One-hot encoding
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    valY = to_categorical(valY)

    return trainX, trainY, testX, testY, valX, valY