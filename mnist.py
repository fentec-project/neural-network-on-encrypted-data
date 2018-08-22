# Copyright (c) 2018 XLAB d.o.o
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# imports
import tensorflow as tf
import numpy as np

img_h = img_w = 28             # MNIST images are 28x28
img_size_flat = img_h * img_w + 1 # 28x28 + 1=785, the total number of pixels plus bias
n_classes = 10                 # Number of classes, one class per digit
disc_data = 3
disc_value = 80


def load_data(mode='train'):
    """
    Function to (download and) load the MNIST data
    :param mode: train or test
    :return: images and the corresponding labels
    """
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    if mode == 'train':
        x_train, y_train, x_valid, y_valid = mnist.train.images, mnist.train.labels, \
                                             mnist.validation.images, mnist.validation.labels
        return x_train, y_train, x_valid, y_valid
    elif mode == 'test':
        x_test, y_test = mnist.test.images, mnist.test.labels
    return x_test, y_test


def randomize(x, y):
    """ Randomizes the order of data samples and their corresponding labels"""
    permutation = np.random.permutation(y.shape[0])
    shuffled_x = x[permutation, :]
    shuffled_y = y[permutation]
    return shuffled_x, shuffled_y


def get_next_batch(x, y, start, end):
    x_batch = x[start:end]
    y_batch = y[start:end]
    return x_batch, y_batch


def discretize_data(d, size):
    max_value = np.amax(np.abs(d))
    res = np.floor((d / max_value) * size) / size
    return res

def add_ones(d):
    ones = np.ones((d.shape[0], 1))
    return np.concatenate([ones, d], axis=1)


# Load MNIST data
x_train, y_train, x_valid, y_valid = load_data(mode='train')

x_train = add_ones(discretize_data(x_train, disc_data))
x_valid = add_ones(discretize_data(x_valid, disc_data))


print("Size of:")
print("- Training-set:\t\t{}".format(len(y_train)))
print("- Validation-set:\t{}".format(len(y_valid)))


print('x_train:\t{}'.format(x_train.shape))
print('y_train:\t{}'.format(y_train.shape))
print('x_train:\t{}'.format(x_valid.shape))
print('y_valid:\t{}'.format(y_valid.shape))

# Hyper-parameters
epochs = 10             # Total number of training epochs
batch_size = 100        # Training batch size
display_freq = 100      # Frequency of displaying the training results
learning_rate = 0.001   # The optimization initial learning rate

h1 = 20       # The first hidden layer is a projection to h1 dimensions

# weight and bais wrappers
def weight_variable(name, shape):
    """
    Create a weight variable with appropriate initialization
    :param name: weight name
    :param shape: weight shape
    :return: initialized weight variable
    """
    initer = tf.truncated_normal_initializer(stddev=0.01)
    return tf.get_variable('W_' + name,
                           dtype=tf.float32,
                           shape=shape,
                           initializer=initer)



# Create the graph for the linear model
# Placeholders for inputs (x) and outputs(y)
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='X')
y = tf.placeholder(tf.float32, shape=[None, n_classes], name='Y')


P = weight_variable('projection', shape=[img_size_flat, h1])
fc1 = tf.matmul(x, P)

x2 = tf.square(fc1)

D = weight_variable('output', shape=[h1, n_classes])
output_logits = tf.matmul(x2, D)

# Network predictions
cls_prediction = tf.argmax(output_logits, axis=1, name='predictions')

# Define the loss function, optimizer, and accuracy
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output_logits), name='loss')
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='Adam-op').minimize(loss)
correct_prediction = tf.equal(tf.argmax(output_logits, 1), tf.argmax(y, 1), name='correct_pred')
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

# Create the op for initializing all variables
init = tf.global_variables_initializer()

sess = tf.InteractiveSession()
sess.run(init)
global_step = 0
# Number of training iterations in each epoch
num_tr_iter = int(len(y_train) / batch_size)
for epoch in range(epochs):
    print('Training epoch: {}'.format(epoch + 1))
    x_train, y_train = randomize(x_train, y_train)
    for iteration in range(num_tr_iter):
        global_step += 1
        start = iteration * batch_size
        end = (iteration + 1) * batch_size
        x_batch, y_batch = get_next_batch(x_train, y_train, start, end)

        # Run optimization op (backprop)
        feed_dict_batch = {x: x_batch, y: y_batch}
        sess.run(optimizer, feed_dict=feed_dict_batch)

        if iteration % display_freq == 0:
            # Calculate and display the batch loss and accuracy
            loss_batch, acc_batch = sess.run([loss, accuracy],
                                             feed_dict=feed_dict_batch)

            print("iter {0:3d}:\t Loss={1:.2f},\tTraining Accuracy={2:.01%}".
                  format(iteration, loss_batch, acc_batch))

    # Run validation after every epoch
    feed_dict_valid = {x: x_valid[:5000], y: y_valid[:5000]}
    Pr, Di, loss_valid, acc_valid = sess.run([P, D, loss, accuracy], feed_dict=feed_dict_valid)
    print('---------------------------------------------------------')
    print("Epoch: {0}, validation loss: {1:.2f}, validation accuracy: {2:.01%}".
          format(epoch + 1, loss_valid, acc_valid))
    print('---------------------------------------------------------')
# print(Di)


Pr = discretize_data(Pr, disc_value)
Di = discretize_data(Di, disc_value)
predict = np.matmul(np.square(np.matmul(x_valid, Pr)), Di)
correct = np.equal(np.argmax(predict, 1), np.argmax(y_valid, 1))
final_acc = np.mean(correct)
print('-----------------------------------------------------------------')
print('-----------------------------------------------------------------')
print('-----------------------------------------------------------------')
print("The final accuracy of validation set after discretization: {0:.01%}".
      format(final_acc))
print('-----------------------------------------------------------------')
print('-----------------------------------------------------------------')
print('-----------------------------------------------------------------')


def matrix_to_txt(Mat, name):
    w = open(name + '.txt', 'w')
    for i in range(Mat.shape[0]):
        row = [str(x) for x in Mat[i, :]]
        w.write(' '.join(row) + '\n')
    w.close()


Pr = Pr * disc_value
Pr = Pr.astype(int)
Di = Di * disc_value
Di = Di.astype(int)
valid1 = np.floor(x_valid[:1] * disc_data + 0.5)
valid1 = valid1.astype(int)

matrix_to_txt(np.transpose(Pr), 'testdata/mat_proj')
matrix_to_txt(np.transpose(Di), 'testdata/mat_diag')
matrix_to_txt(valid1, 'testdata/mat_valid')