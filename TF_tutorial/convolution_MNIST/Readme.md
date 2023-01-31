# Vitrual Environment

* Tensorflow 2.2.0 + gpu

* Python 3.7.13


# Features about TensorFlow 2.0

* Eager Execution

* Discard "session"

* @tf.function

* tf.keras

## Process for training

1. Prepare the input data
2. Build the model
3. Define the loss function and optimizer
4. Run the model

## Difference of input channel between Pytorch and TensorFlow

* Pytorch: [batch, channel, height, width]

* TensorFlow (`data_format`: default `channel_last`): [batch, height, width, channel]

* TensorFlow (`data_format`: `channel_first`): [batch, channels, height, width]

## Padding

* valid (don't use padding): N = ceiling((W - F + 1) / S)

* SAME (using padding): N = ceiling(W / S)

**Image size** = M * M, **Filter size** = F * F, **Stride** = S

## Loss

* SparseCategoricalCrossentropy: for y = [0, 1, 2 ...] integer for category

* CategoricalCrossentropy: for y = onehot encode