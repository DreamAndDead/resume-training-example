import os
import pickle
import json
import argparse
import keras
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.callbacks import Callback
from keras.models import load_model
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")


class DataLoader:
    def load_dataset(self):
        (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

        X = np.vstack((train_images, test_images))
        Y = np.hstack((train_labels, test_labels))

        X = self.preprocess(X)

        self.le = LabelBinarizer()
        Y = self.le.fit_transform(Y)
    
        return train_test_split(X, Y, test_size=0.2, random_state=1)

    def preprocess(self, X):
        return X.reshape(-1, 28, 28, 1).astype('float32') / 255.0

    def save_label_encoder(self, le_file):
        pickle.dump(self.le, open(le_file, 'wb'))

    def load_label_encoder(self, le_file):
        return pickle.load(open(le_file, 'rb'))


def create_model():
    model = keras.models.Sequential([
        keras.layers.convolutional.Conv2D(16, (2, 2), strides=(1, 1), padding='same', input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Activation('relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer=keras.optimizers.SGD(lr=0.005),
                  loss=keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])

    return model

class TrainingState(Callback):
    def __init__(self, state_dir):
        super(TrainingState, self).__init__()

        self.state_dir = state_dir

        self.json_log_file = os.path.join(self.state_dir, 'logs.json')
        self.png_log_file = os.path.join(self.state_dir, 'logs.png')
        self.last_model_file = os.path.join(self.state_dir, 'last_model.h5')
        self.best_model_file = os.path.join(self.state_dir, 'best_model.h5')

        if os.path.exists(self.json_log_file):
            with open(self.json_log_file, 'r') as f:
                self.history = json.load(f)
        else:
            self.history = {
                'epoch': -1,
                'best': 0
            }

    def get_initial_epoch(self):
        return self.history['epoch'] + 1

    def load_last_model(self):
        if os.path.exists(self.last_model_file):
            return load_model(self.last_model_file)
        else:
            return None

    def on_epoch_end(self, epoch, logs=None):
        self.history['epoch'] = epoch
        
        logs = logs or None
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        self.save_json_log()
        self.save_png_log()
        self.save_last_model()
        self.save_best_model()

    def save_json_log(self):
        with open(self.json_log_file, 'w') as f:
            json.dump(self.history, f)

        print('save json log to {}'.format(self.json_log_file))

    def save_png_log(self):
        history = self.history
        size = history['epoch'] + 1
        
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(0, size), history["loss"], label="train_loss")
        plt.plot(np.arange(0, size), history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, size), history["acc"], label="train_acc")
        plt.plot(np.arange(0, size), history["val_acc"], label="val_acc")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend()
        plt.savefig(self.png_log_file)

        print('save png log to {}'.format(self.png_log_file))

    def save_last_model(self):
        self.model.save(self.last_model_file)
        print('save last model to {}'.format(self.last_model_file))

    def save_best_model(self):
        epoch = self.history['epoch']
        best = self.history['best']
        val_acc = self.history['val_acc']
        
        if val_acc[-1] > best:
            self.history['best'] = val_acc[-1]
            self.model.save(self.best_model_file)
            print('val_acc inc from {} to {}, save best model to {}'.format(best, val_acc[-1], self.best_model_file))
        else:
            print('no inc in val_acc ...')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", required=True,
                    help="output dir to save training state")
    args = vars(ap.parse_args())

    output_dir = args['output']
    os.makedirs(output_dir, exist_ok=True)
    label_encoder_file = os.path.join(output_dir, 'label_encoder.pkl')

    data_loader = DataLoader()
    trainX, testX, trainY, testY = data_loader.load_dataset()
    data_loader.save_label_encoder(label_encoder_file)

    trainingState = TrainingState(output_dir)
    
    model = trainingState.load_last_model()
    if not model:
        model = create_model()
        
    model.summary()

    model.fit(trainX, trainY,
              validation_data=(testX, testY),
              epochs=10, batch_size=32,
              callbacks=[trainingState],
              initial_epoch=trainingState.get_initial_epoch())

    predictions = model.predict(testX, batch_size=64)
    report = classification_report(testY.argmax(axis=1),
                                   predictions.argmax(axis=1))
    print(report)
