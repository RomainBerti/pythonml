# Imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
    """Model function for CNN.
     Args:
        features:  A dict containing the features passed to the model with train_input_fn in
           training mode, with eval_input_fn in evaluation mode, and with serving_input_fn
           in predict mode.
        labels:    Tensor containing the labels passed to the model with train_input_fn in
           training mode and eval_input_fn in evaluation mode. It is empty for
           predict mode.
        mode:     One of the following tf.estimator.ModeKeys string values indicating the context
           in which the model_fn was invoked:
           - TRAIN: The model_fn was invoked in training mode.
           - EVAL: The model_fn was invoked in evaluation mode.
           - PREDICT: The model_fn was invoked in predict mode:

    Returns: An EstimatorSpec, which contains evaluation and loss function. """
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    #  Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def train_input_fn(training_dir):
    """
    Implement code to do the following:
    1. Read the **training** dataset files located in training_dir
    2. Preprocess the dataset
    3. Return 1) a mapping of feature columns to Tensors with
    the corresponding feature data, and 2) a Tensor containing labels

    For more information on how to create a input_fn, see https://www.tensorflow.org/get_started/input_fn.

    Args:
        training_dir:    Directory where the dataset is located inside the container.

    Returns: (data, labels) tuple
    """


def eval_input_fn(training_dir):
    """
   Implement code to do the following:
    1. Read the **evaluation** dataset files located in training_dir
    2. Preprocess the dataset
    3. Return 1) a mapping of feature columns to Tensors with
    the corresponding feature data, and 2) a Tensor containing labels

    For more information on how to create a input_fn, see https://www.tensorflow.org/get_started/input_fn.

    Args:
     training_dir: The directory where the dataset is located inside the container.

    Returns: (data, labels) tuple
    """


def serving_input_fn(hyperparameters):
    """
    During training, a train_input_fn() ingests data and prepares it for use by the model.
    At the end of training, similarly, a serving_input_fn() is called to create the model that
    will be exported for Tensorflow Serving.

    Use this function to do the following:

        - Add placeholders to the graph that the serving system will feed with inference requests.
        - Add any additional operations needed to convert data from the input format into the
        feature Tensors expected by the model.

    The function returns a tf.estimator.export.ServingInputReceiver object, which packages the placeholders
      and the resulting feature Tensors together.

    Typically, inference requests arrive in the form of serialized tf.Examples, so the
    serving_input_receiver_fn() creates a single string placeholder to receive them. The serving_input_receiver_fn()
    is then also responsible for parsing the tf.Examples by adding a tf.parse_example operation to the graph.

    For more information on how to create a serving_input_fn, see
      https://github.com/tensorflow/tensorflow/blob/18003982ff9c809ab8e9b76dd4c9b9ebc795f4b8/tensorflow/docs_src/programmers_guide/saved_model.md#preparing-serving-inputs.

    Args:
    Returns:

	"""

def main(unused_argv):
    # Load training and eval data
    # mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    # train_data = mnist.train.images  # Returns np.array
    # print(len(train_data))
    # train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    # eval_data = mnist.test.images  # Returns np.array
    # print(len(eval_data))
    # eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    # # Create the Estimator
    # mnist_classifier = tf.estimator.Estimator(
    #     model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")
    # # Set up logging for predictions
    # tensors_to_log = {"probabilities": "softmax_tensor"}
    # logging_hook = tf.train.LoggingTensorHook(
    #     tensors=tensors_to_log, every_n_iter=50)
    # # Train the model
    # train_input_fn = tf.estimator.inputs.numpy_input_fn(
    #     x={"x": train_data},
    #     y=train_labels,
    #     batch_size=100,
    #     num_epochs=None,
    #     shuffle=True)
    # mnist_classifier.train(
    #     input_fn=train_input_fn,
    #     steps=2000,
    #     hooks=[logging_hook])
    # # Evaluate the model and print results
    # eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    #     x={"x": eval_data},
    #     y=eval_labels,
    #     num_epochs=1,
    #     shuffle=False)
    # eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    # print(eval_results)

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib

    # mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    # train_data = mnist.train.images
    # print(train_data)
    # pixels = np.array(train_data, dtype='uint8')
    # # MN_train = np.loadtxt( ... )
    # for i in range(0, 2):
    #     #(np.reshape(arr, (28, 28)) * 255).astype(np.uint8)
    #     grey = (train_data[0+784*i:784+784*i, 0].reshape(28, 28)*255)
    #     #print(grey)
    #     plt.imshow(grey, cmap=matplotlib.cm.binary)
    #     plt.show()


    # from matplotlib import pyplot as plt
    # import numpy as np
    # from tensorflow.examples.tutorials.mnist import input_data
    # mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    #
    # def gen_image(arr):
    #     two_d = (np.reshape(arr, (28, 28)) * 255).astype(np.uint8)
    #     plt.imshow(two_d, interpolation='nearest')
    #     return plt
    #
    # # Get a batch of two random images and show in a pop-up window.
    # batch_xs, batch_ys = mnist.test.next_batch(2)
    # gen_image(batch_xs[0]).show()
    # gen_image(batch_xs[1]).show()



if __name__ == "__main__":
    tf.app.run()

