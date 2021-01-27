import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, AveragePooling2D, Dropout, Conv1D, AveragePooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
import datetime
import numpy as np
from tensorflow_privacy.privacy.optimizers import dp_optimizer
from tensorflow_privacy.privacy.analysis import privacy_ledger
from tensorflow_privacy.privacy.dp_query import gaussian_query
from absl import logging
import collections
import logging
from scipy import stats
import seaborn as sns
sns.set()

BATCH_SIZE = 128
EPOCHS = 25
LOGDIR = "./logs/"

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

logger = tf.get_logger()
logger.setLevel(logging.ERROR)


def make_optimizer_class(cls):
    """Constructs a DP optimizer class from an existing one."""
    parent_code = tf.compat.v1.train.Optimizer.compute_gradients.__code__
    child_code = cls.compute_gradients.__code__
    GATE_OP = tf.compat.v1.train.Optimizer.GATE_OP  # pylint: disable=invalid-name
    if child_code is not parent_code:
        logging.warning(
            'WARNING: Calling make_optimizer_class() on class %s that overrides '
            'method compute_gradients(). Check to ensure that '
            'make_optimizer_class() does not interfere with overridden version.',
            cls.__name__)

    class DPOptimizerClass(cls):
        """Differentially private subclass of given class cls."""

        _GlobalState = collections.namedtuple(
            '_GlobalState', ['l2_norm_clip', 'stddev'])

        def __init__(
                self,
                dp_sum_query,
                num_microbatches=None,
                unroll_microbatches=False,
                *args,  # pylint: disable=keyword-arg-before-vararg, g-doc-args
                **kwargs):
            """Initialize the DPOptimizerClass.

            Args:
              dp_sum_query: DPQuery object, specifying differential privacy
                mechanism to use.
              num_microbatches: How many microbatches into which the minibatch is
                split. If None, will default to the size of the minibatch, and
                per-example gradients will be computed.
              unroll_microbatches: If true, processes microbatches within a Python
                loop instead of a tf.while_loop. Can be used if using a tf.while_loop
                raises an exception.
            """
            super(DPOptimizerClass, self).__init__(*args, **kwargs)
            self._dp_sum_query = dp_sum_query
            self._num_microbatches = num_microbatches
            self._global_state = self._dp_sum_query.initial_global_state()
            # TODO(b/122613513): Set unroll_microbatches=True to avoid this bug.
            # Beware: When num_microbatches is large (>100), enabling this parameter
            # may cause an OOM error.
            self._unroll_microbatches = unroll_microbatches

        def compute_gradients(self,
                              loss,
                              var_list,
                              gate_gradients=GATE_OP,
                              aggregation_method=None,
                              colocate_gradients_with_ops=False,
                              grad_loss=None,
                              gradient_tape=None,
                              curr_noise_mult=0,
                              curr_norm_clip=1):

            self._dp_sum_query = gaussian_query.GaussianSumQuery(curr_norm_clip,
                                                                 curr_norm_clip * curr_noise_mult)
            self._global_state = self._dp_sum_query.make_global_state(curr_norm_clip,
                                                                      curr_norm_clip * curr_noise_mult)

            # TF is running in Eager mode, check we received a vanilla tape.
            if not gradient_tape:
                raise ValueError('When in Eager mode, a tape needs to be passed.')

            vector_loss = loss()
            if self._num_microbatches is None:
                self._num_microbatches = tf.shape(input=vector_loss)[0]
            sample_state = self._dp_sum_query.initial_sample_state(var_list)
            microbatches_losses = tf.reshape(vector_loss, [self._num_microbatches, -1])
            sample_params = (self._dp_sum_query.derive_sample_params(self._global_state))

            def process_microbatch(i, sample_state):
                """Process one microbatch (record) with privacy helper."""
                microbatch_loss = tf.reduce_mean(input_tensor=tf.gather(microbatches_losses, [i]))
                grads = gradient_tape.gradient(microbatch_loss, var_list)
                sample_state = self._dp_sum_query.accumulate_record(sample_params, sample_state, grads)
                return sample_state

            for idx in range(self._num_microbatches):
                sample_state = process_microbatch(idx, sample_state)

            if curr_noise_mult > 0:
                grad_sums, self._global_state = (self._dp_sum_query.get_noised_result(sample_state, self._global_state))
            else:
                grad_sums = sample_state

            def normalize(v):
                return v / tf.cast(self._num_microbatches, tf.float32)

            final_grads = tf.nest.map_structure(normalize, grad_sums)
            grads_and_vars = final_grads  # list(zip(final_grads, var_list))

            return grads_and_vars

    return DPOptimizerClass


def make_gaussian_optimizer_class(cls):
    """Constructs a DP optimizer with Gaussian averaging of updates."""

    class DPGaussianOptimizerClass(make_optimizer_class(cls)):
        """DP subclass of given class cls using Gaussian averaging."""

        def __init__(
                self,
                l2_norm_clip,
                noise_multiplier,
                num_microbatches=None,
                ledger=None,
                unroll_microbatches=False,
                *args,  # pylint: disable=keyword-arg-before-vararg
                **kwargs):
            dp_sum_query = gaussian_query.GaussianSumQuery(
                l2_norm_clip, l2_norm_clip * noise_multiplier)

            if ledger:
                dp_sum_query = privacy_ledger.QueryWithLedger(dp_sum_query,
                                                              ledger=ledger)

            super(DPGaussianOptimizerClass, self).__init__(
                dp_sum_query,
                num_microbatches,
                unroll_microbatches,
                *args,
                **kwargs)

        @property
        def ledger(self):
            return self._dp_sum_query.ledger

    return DPGaussianOptimizerClass


class LeNet(Sequential):
    def __init__(self, dataset_name, dropout=False, dp=False, lr=0.01):
        super().__init__()
        if dataset_name == "MNIST":
            input_shape = (28, 28)
            num_classes = 10
        elif dataset_name == "CIFAR-10":
            input_shape = (32, 32, 3)
            num_classes = 10
        else:
            print("Only MNIST and CIFAR-10 are acceptable datasets at this time.")
            exit(1)
        if dropout:
            self.add(Dropout(0.05))
        if dataset_name == "MNIST":
            self.add(Conv1D(6, kernel_size=5, activation='relu', input_shape=input_shape))
            self.add(AveragePooling1D())
        elif dataset_name == "CIFAR-10":
            self.add(Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
            self.add(AveragePooling2D())
        if dropout:
            self.add(Dropout(0.05))
        if dataset_name == "MNIST":
            self.add(Conv1D(16, kernel_size=5, activation='relu', input_shape=input_shape))
            self.add(AveragePooling1D())
        elif dataset_name == "CIFAR-10":
            self.add(Conv2D(16, kernel_size=(5, 5), activation='relu'))
            self.add(AveragePooling2D())
        self.add(Flatten())
        if dropout:
            self.add(Dropout(0.05))
        self.add(Dense(120, activation='relu'))
        if dropout:
            self.add(Dropout(0.05))
        self.add(Dense(84, activation='relu'))
        if dropout:
            self.add(Dropout(0.05))
        self.add(Dense(num_classes, activation='softmax'))

        if dp is True:
            l2_norm_clip = 1.5
            noise_multiplier = 1.3
            num_microbatches = BATCH_SIZE

            GradientDescentOptimizer = tf.compat.v1.train.GradientDescentOptimizer
            DPGradientDescentGaussianOptimizer_NEW = make_gaussian_optimizer_class(GradientDescentOptimizer)

            opt = DPGradientDescentGaussianOptimizer_NEW(
                l2_norm_clip=l2_norm_clip,
                noise_multiplier=noise_multiplier,
                num_microbatches=num_microbatches,
                learning_rate=lr)
        else:
            opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.0)


        self.compile(optimizer=opt, loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])


if __name__ == "__main__":
    # Instantiate models for CIFAR-10
    print("Instantiating models for CIFAR-10...")
    CIFAR_vanilla = LeNet("CIFAR-10")
    dp_CIFAR = LeNet("CIFAR-10", dp=True)
    dropout_CIFAR = LeNet("CIFAR-10", dropout=True)
    dp_dropout_CIFAR = LeNet("CIFAR-10", dropout=True, dp=True)

    modeldict = {"LeNet5": CIFAR_vanilla,
                 "DP_LeNet": dp_CIFAR,
                 "dropout_LeNet": dropout_CIFAR,
                 "DP_dropout_LeNet": dp_dropout_CIFAR}

    # Load CIFAR data
    print("Loading CIFAR-10 data...")
    train, test = tf.keras.datasets.cifar10.load_data()
    train_data, train_labels = train
    test_data, test_labels = test

    train_data = np.array(train_data, dtype=np.float32) / 255
    test_data = np.array(test_data, dtype=np.float32) / 255

    train_data = train_data.reshape(train_data.shape[0], 32, 32, 3)
    test_data = test_data.reshape(test_data.shape[0], 32, 32, 3)

    train_labels = np.array(train_labels, dtype=np.int32)
    test_labels = np.array(test_labels, dtype=np.int32)

    train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=10)
    test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)

    for name, model in modeldict.items():
        TBLOGDIR = LOGDIR + "CIFAR-10_" + name
        tensorboard_callback = TensorBoard(log_dir=TBLOGDIR, histogram_freq=1)
        model.fit(train_data, train_labels,
                  epochs=EPOCHS,
                  validation_data=(test_data, test_labels),
                  batch_size=BATCH_SIZE,
                  callbacks=[tensorboard_callback])

    # Instantiate models for MNIST
    print("Instantiating models for MNIST...")
    MNIST_vanilla = LeNet("MNIST")
    dp_MNIST = LeNet("MNIST", dp=True)
    dropout_MNIST = LeNet("MNIST", dropout=True)
    dp_dropout_MNIST = LeNet("MNIST", dropout=True, dp=True)

    modeldict = {"LeNet5": MNIST_vanilla,
                 "DP_LeNet": dp_MNIST,
                 "dropout_LeNet": dropout_MNIST,
                 "DP_dropout_LeNet": dp_dropout_MNIST}

    # Load MNIST data
    print("Loading MNIST data...")
    train, test = tf.keras.datasets.mnist.load_data()
    train_data, train_labels = train
    test_data, test_labels = test

    train_data = np.array(train_data, dtype=np.float32) / 255
    test_data = np.array(test_data, dtype=np.float32) / 255

    train_data = train_data.reshape(train_data.shape[0], 28, 28)
    test_data = test_data.reshape(test_data.shape[0], 28, 28)

    train_labels = np.array(train_labels, dtype=np.int32)
    test_labels = np.array(test_labels, dtype=np.int32)

    train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=10)
    test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)

    for name, model in modeldict.items():
        TBLOGDIR = LOGDIR + "MNIST_" + name
        tensorboard_callback = TensorBoard(log_dir=TBLOGDIR, histogram_freq=1)
        model.fit(train_data, train_labels,
                  epochs=EPOCHS,
                  validation_data=(test_data, test_labels),
                  batch_size=BATCH_SIZE,
                  callbacks=[tensorboard_callback])