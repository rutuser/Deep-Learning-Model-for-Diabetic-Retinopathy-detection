from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalMaxPool2D
from tensorflow.keras.applications import VGG19
from tensorflow.keras.optimizers import Adadelta
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


class Model:
    def __init__(self, INPUT_SHAPE) -> None:
        base_model = VGG19(include_top=False,
                           weights="imagenet",
                           input_shape=INPUT_SHAPE)

        # FINE TUNING
        trainable = True
        base_model.trainable = trainable

        for layer in base_model.layers:
            if layer.name.startswith('block2'):
                break
            layer.trainable = False
            print('Layer ' + layer.name + ' frozen...')

        # TOP MODEL ARCHITECTURE
        model = Sequential()
        model.add(base_model)
        model.add(GlobalMaxPool2D())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(5, activation='sigmoid'))

        self.model = model

    def architecture(self) -> None:
        print(self.model.summary())

    def compile(self) -> None:
        self.model.compile(loss="binary_crossentropy",
                           optimizer=Adadelta(learning_rate=0.001),
                           metrics=["accuracy"])

    def train(self, trainX, trainY, valX, valY, batch_size, epochs):
        H = self.model.fit(trainX,
                             trainY,
                             steps_per_epoch=len(trainX) / batch_size,
                             validation_data=(valX, valY),
                             validation_steps=len(valX) / batch_size,
                             batch_size=batch_size,
                             epochs=epochs,
                             verbose=1)

        self.history = H.history

    def show_history(self) -> None:
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(1, len(self.history['loss'])), self.history["accuracy"][1:], label="train_acc")
        plt.plot(np.arange(1, len(self.history['loss'])), self.history["val_accuracy"][1:], label="val_acc")
        plt.title("Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Train_ACC / Val_ACC")
        plt.legend()

        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(1, len(self.history['loss'])), self.history["loss"][1:], label="train_loss")
        plt.plot(np.arange(1, len(self.history['loss'])), self.history["val_loss"][1:], label="val_loss")
        plt.title("Loss")
        plt.xlabel("Epoch #")
        plt.ylabel("Train_Loss / Val_Loss")
        plt.legend()
    
    def evalutate(self, testX, testY) -> None:
        pred = self.model.predict(testX, batch_size=128)
        pred_classes = np.argmax(pred, axis=1)
        expected_clases = np.argmax(testY, axis=1) # remove one-hot encoding
        acc = accuracy_score(pred_classes, expected_clases)

        # Precision en prediccion
        print(f'Accuracy {acc}')
        # Matriz de confusión
        print(confusion_matrix(expected_clases, pred.argmax(axis=1)))
        # Evaluando el modelo de predicción con las imágenes de test
        print(classification_report(expected_clases, pred.argmax(axis=1)))

    def save(self, target_path) -> None:
        self.model.save_weights(target_path)