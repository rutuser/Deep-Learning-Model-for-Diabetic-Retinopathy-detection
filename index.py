from model import Model
from loadFromCSV import loadFromCSV
from processData import processData

INPUT_SHAPE = (80,80,3)
X ,Y = loadFromCSV(INPUT_SHAPE)
print(X.shape)
print(Y.shape)
trainX, trainY, testX, testY, valX, valY = processData(X, Y)

model_A = Model(INPUT_SHAPE)
model_A.compile()
model_A.train(trainX, trainY, valX, valY, batch_size=64, epochs=500)
model_A.show_history()
model_A.save('sigmoid_model1.h5')
model_A.evalutate(testX, testY)
