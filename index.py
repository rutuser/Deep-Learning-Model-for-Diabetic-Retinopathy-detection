from model import Model
from loadImages import loadImages
from processData import processData

INPUT_SHAPE = (80,80,3)
X ,Y = loadImages()
trainX, trainY, testX, testY, valX, valY = processData(X, Y)

model = Model(INPUT_SHAPE)
model.compile()
model.train(trainX, trainY, valX, valY, batch_size=64, epochs=20)
model.show_history()
model.save()
model.evalutate(testX, testY)

