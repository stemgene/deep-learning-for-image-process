from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from model import MyModel
import numpy as np
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# # Check the images
# imgs = x_test[:3]
# labs = y_test[:3]
# print(labs)
# plot_imgs = np.hstack(imgs)
# plt.imshow(plot_imgs, cmap='gray')
# plt.show()

# Add a channels dimension [60000, 28, 28] --> [60000, 28, 28, 1]
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# create data generator
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)  # 10000代表读入内存的图片，越接近原始数据大小，越接近随机采样，但受内存的制约
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# define loss
model = MyModel()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

# define optimizer
optimizer = tf.keras.optimizers.Adam()

# define train_loss and train_accuracy
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

# define test_loss and test_accuracy
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

# define train function including calculating loss, applying gradient and calculating accuracy
@tf.function
def train_step(image, labels):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different behavior during training versus inference (e.g. Dropout).
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    train_loss(loss)
    train_accuracy(labels, predictions)

# define test funtion including claculating loss and calculating accuracy
@tf.function
def test_step(images, labels):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)
    
    test_loss(t_loss)
    test_accuracy(labels, predictions)
    
EPOCHS = 5

for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()
    
    for images, labels in train_ds:
        train_step(images, labels)
    
    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)
    
    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss {}, Test Accuracy: {}'
    print(template.format(epoch + 1, 
                        train_loss.result(),
                        train_accuracy.result() * 100,
                        test_loss.result(),
                        test_accuracy.result() * 100))