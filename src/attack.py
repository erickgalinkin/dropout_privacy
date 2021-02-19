import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import logging

EPOCHS = 25
LOGDIR = "./logs/"
MODEL_PATH = "./model/"
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

logger = tf.get_logger()
logger.setLevel(logging.ERROR)


def random_sample(data):
    idx = [np.random.randint(len(data)) for _ in data[1]]
    x = data[0][idx]
    y = data[1][idx]
    return (x, y)


class VGGNet(Sequential):
    def __init__(self, input_shape):
        super().__init__()
        num_classes = 10
        if len(input_shape) == 2:
            self.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape))
            self.add(MaxPooling1D())
            self.add(Conv1D(64, kernel_size=3, activation='relu'))
            self.add(MaxPooling1D())
        elif len(input_shape) == 3:
            self.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
            self.add(MaxPooling2D())
            self.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
            self.add(MaxPooling2D())
        self.add(Flatten())
        self.add(Dense(120, activation='relu'))
        self.add(Dense(84, activation='relu'))
        self.add(Dense(num_classes, activation='softmax'))

        opt = tf.keras.optimizers.Adam()

        self.compile(optimizer=opt, loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])


class AttackNet(Sequential):
    def __init__(self):
        super().__init__()
        self.add(Dense(64, activation='relu', input_shape=(10,)))
        self.add(Dense(64, activation='relu'))
        self.add(Dense(1, activation='sigmoid'))

        opt = tf.keras.optimizers.Adam()

        self.compile(optimizer=opt, loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])


def train_shadow_model(train, test, shape):
    # perm_train = random_sample(train)
    # perm_test = random_sample(test)
    train_data, train_labels = train
    test_data, test_labels = test

    train_data = np.array(train_data, dtype=np.float32) / 255
    test_data = np.array(test_data, dtype=np.float32) / 255
    train_labels = np.array(train_labels, dtype=np.int32)
    test_labels = np.array(test_labels, dtype=np.int32)

    train_data = train_data.reshape(train_data.shape[0], shape[0], shape[1], shape[2])
    test_data = test_data.reshape(test_data.shape[0], shape[0], shape[1], shape[2])
    train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=10)
    test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)

    shadow_model = VGGNet(shape)
    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=3)
    TBLOGDIR = LOGDIR + "Shadow"
    tensorboard_callback = TensorBoard(log_dir=TBLOGDIR, histogram_freq=1)
    history = shadow_model.fit(train_data, train_labels,
                               batch_size=100,
                               epochs=EPOCHS,
                               validation_data=(test_data, test_labels),
                               verbose=1,
                               callbacks=[earlystopping, tensorboard_callback])

    print(f"Shadow model training accuracy: {history.history['accuracy']}\n "
          f"Shadow model validation accuracy:{history.history['val_accuracy']}\n")

    return train_data, test_data, shadow_model


def train_attack_model(train_x, test_x, shadow):
    attack_model = AttackNet()
    in_label = np.ones(len(train_x))
    out_label = np.zeros(len(test_x))
    in_preds = shadow.predict(train_x)
    out_preds = shadow.predict(test_x)

    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=2)
    TBLOGDIR = LOGDIR + "Attack"
    tensorboard_callback = TensorBoard(log_dir=TBLOGDIR, histogram_freq=1)
    attack_model.fit(in_preds, in_label,
                     epochs=EPOCHS,
                     validation_data=(out_preds, out_label),
                     verbose=0,
                     callbacks=[earlystopping, tensorboard_callback])
    return attack_model


def attack_target(attack_model, target, train, test):
    train_data, train_labels = train
    test_data, test_labels = test

    target_model = tf.keras.models.load_model(MODEL_PATH + target)

    preds = target_model.predict(train_data)
    in_preds = attack_model.predict(preds)
    in_preds = [1 if p >= 0.5 else 0 for p in in_preds]
    in_acc = np.average(in_preds)

    preds = target_model.predict(test_data)
    out_preds = attack_model.predict(preds)
    out_preds = [1 if p >= 0.5 else 0 for p in out_preds]
    out_acc = np.average(out_preds)
    overall = 0.5 * (in_acc + out_acc)
    print("Training set attack accuracy for {}: {}".format(target, in_acc))
    print("Test set attack accuracy for {}: {}".format(target, out_acc))
    print("Overall attack accuracy for {}: {}".format(target, overall))


if __name__ == "__main__":
    models = ["LeNet5", "DP_LeNet", "dropout_LeNet", "DP_dropout_LeNet"]
    cifar_models = ["CIFAR-10_" + name for name in models]
    mnist_models = ["MNIST_" + name for name in models]

    cifar_train, cifar_test = tf.keras.datasets.cifar10.load_data()
    train_x, test_x, shadow = train_shadow_model(cifar_train, cifar_test, (32, 32, 3))
    attack = train_attack_model(train_x, test_x, shadow)
    for model in cifar_models:
        attack_target(attack, model, cifar_train, cifar_test)
