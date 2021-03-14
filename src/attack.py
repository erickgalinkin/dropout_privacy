import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import logging
import matplotlib.pyplot as plt

EPOCHS = 25
LOGDIR = "./logs/"
MODEL_PATH = "./model/"
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

logger = tf.get_logger()
logger.setLevel(logging.ERROR)


class VGGNet(Sequential):
    def __init__(self, input_shape):
        super().__init__()
        num_classes = 10
        self.add(self.preprocessing())
        if len(input_shape) == 2:
            self.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape))
            self.add(MaxPooling2D())
            self.add(Conv1D(64, kernel_size=3, activation='relu'))
            self.add(MaxPooling2D())
        elif len(input_shape) == 3:
            self.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
            self.add(MaxPooling2D())
            self.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
            self.add(MaxPooling2D())
        self.add(Flatten())
        self.add(Dense(120, activation='relu'))
        self.add(Dense(84, activation='relu'))
        self.add(Dense(num_classes, activation='softmax'))

        opt = tf.keras.optimizers.SGD()

        self.compile(optimizer=opt, loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])

    @staticmethod
    def preprocessing():
        IMG_SIZE = 32
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.25),
            tf.keras.layers.experimental.preprocessing.Resizing(IMG_SIZE, IMG_SIZE),
            tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)
        ])
        return data_augmentation


class AttackNet(Sequential):
    def __init__(self):
        super().__init__()
        self.add(Dense(128, activation='tanh', input_shape=(10,)))
        self.add(Dense(128, activation='tanh'))
        self.add(Dense(1, activation='sigmoid'))

        opt = tf.keras.optimizers.SGD(lr=0.001)

        self.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])


def train_shadow_model(train, test, shape):
    train_data, train_labels = train
    test_data, test_labels = test

    train_labels = np.array(train_labels, dtype=np.int32)
    test_labels = np.array(test_labels, dtype=np.int32)

    if len(shape) == 3:
        train_data = train_data.reshape(train_data.shape[0], shape[0], shape[1], shape[2])
        test_data = test_data.reshape(test_data.shape[0], shape[0], shape[1], shape[2])
    else:
        train_data = train_data.reshape(train_data.shape[0], shape[0], shape[1], 1)
        test_data = test_data.reshape(test_data.shape[0], shape[0], shape[1], 1)

    train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=10)
    test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)

    shadow_model = VGGNet(shape)
    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=3)
    TBLOGDIR = LOGDIR + "Shadow"
    tensorboard_callback = TensorBoard(log_dir=TBLOGDIR, histogram_freq=1)
    history = shadow_model.fit(train_data, train_labels,
                               epochs=EPOCHS,
                               validation_data=(test_data, test_labels),
                               verbose=1,
                               callbacks=[earlystopping, tensorboard_callback])

    return train_data, test_data, shadow_model


def train_attack_model(train_x, test_x, shadow, cutoff):
    print("Training attack model...")
    attack_model = AttackNet()
    in_label = np.ones(len(train_x))
    out_label = np.zeros(len(test_x))
    in_preds = shadow.predict(train_x)
    out_preds = shadow.predict(test_x)

    preds = np.concatenate((in_preds, out_preds))
    labels = np.concatenate((in_label, out_label))

    # Need to reduce size of the dataset to avoid data imbalance, so we chop positive class samples off.
    preds = preds[cutoff:]
    labels = labels[cutoff:]

    train_data, test_data, train_label, test_label = train_test_split(preds, labels, train_size=0.95, shuffle=True)

    earlystopping = EarlyStopping(monitor='loss', min_delta=0.0001, patience=5)
    TBLOGDIR = LOGDIR + "Attack"
    tensorboard_callback = TensorBoard(log_dir=TBLOGDIR, histogram_freq=1)
    attack_model.fit(train_data, train_label,
                     epochs=EPOCHS,
                     validation_data=(test_data, test_label),
                     verbose=1,
                     callbacks=[earlystopping, tensorboard_callback])
    return attack_model


def attack_target(attack_model, target, train, test):
    train_data, train_labels = train
    test_data, test_labels = test

    target_model = tf.keras.models.load_model(MODEL_PATH + target)

    preds = target_model.predict(train_data)
    in_preds = attack_model.predict(preds)
    in_preds = [1 if p >= 0.5 else 0 for p in in_preds]

    preds = target_model.predict(test_data)
    out_preds = attack_model.predict(preds)
    # We mark it correct if it predicts the 0 class for the test set data
    out_preds = [1 if p < 0.5 else 0 for p in out_preds]
    overall = np.average(in_preds + out_preds)
    preds = in_preds + out_preds
    labels = [1] * len(in_preds) + [0] * len(out_preds)
    print("Overall attack accuracy for {}: {}".format(target, overall))
    return preds, labels


def plot_auc(results, dataset):
    for k, v in results.items():
        preds = v[0]
        labels = v[1]
        model_name = v[2]
        fpr, tpr, thresholds = roc_curve(labels, preds)
        # Calculate Area under the curve to display on the plot
        auc = roc_auc_score(labels, preds)
        # Now, plot the computed values
        plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (model_name, auc))
    # Custom settings for the plot
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1-Specificity(False Positive Rate)')
    plt.ylabel('Sensitivity(True Positive Rate)')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig("imgs/AUC_" + dataset + ".png")  # Display
    plt.clf()


if __name__ == "__main__":
    models = ["LeNet5", "DP_LeNet", "dropout_LeNet", "DP_dropout_LeNet"]
    cifar_models = ["CIFAR-10_" + name for name in models]
    mnist_models = ["MNIST_" + name for name in models]

    cifar_result_dict = dict()
    cifar_train, cifar_test = tf.keras.datasets.cifar10.load_data()
    train_x, test_x, shadow = train_shadow_model(cifar_train, cifar_test, (32, 32, 3))
    attack = train_attack_model(train_x, test_x, shadow, 40000)
    for model in cifar_models:
        preds, labels = attack_target(attack, model, cifar_train, cifar_test)
        cifar_result_dict[model] = (preds, labels, model)
    plot_auc(cifar_result_dict, "CIFAR-10")

    mnist_result_dict = dict()
    mnist_train, mnist_test = tf.keras.datasets.mnist.load_data()
    train_x, test_x, shadow = train_shadow_model(mnist_train, mnist_test, (28, 28))
    attack = train_attack_model(train_x, test_x, shadow, 50000)
    for model in mnist_models:
        preds, labels = attack_target(attack, model, mnist_train, mnist_test)
        mnist_result_dict[model] = (preds, labels, model)
    plot_auc(mnist_result_dict, "MNIST")
