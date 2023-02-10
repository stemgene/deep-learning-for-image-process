from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from model import AlexNet_v1, AlexNet_v2
import tensorflow as tf
import json
import os

data_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
image_path = data_root + "/data_set/flower_data/"
train_dir = image_path + "train"
validation_dir = image_path + "val"

# create the direction for saving weights
if not os.path.exists("save_weights"):
    os.makedirs("save_weights")
    
im_height = 224
im_width = 224
batch_size = 32
epochs = 3

# data generator with data augmentation
train_image_generator = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
validation_image_generator = ImageDataGenerator(rescale=1./255)

# load data and return a description: Found 3306 images belonging to 5 classes. 
train_data_gen = train_image_generator.flow_from_directory(directory=train_dir, 
                                                            batch_size=batch_size,
                                                            shuffle=True,
                                                            target_size=(im_height, im_width),
                                                            class_mode='categorical') 
# get train set size
total_train = train_data_gen.n #total_train = 3306

# get class dict
class_indices = train_data_gen.class_indices

# transform value and key of dict: {0: 'daisy', 1: 'dandelion', 2: 'roses', 3: 'sunflowers', 4: 'tulips'}
inverse_dict = dict((val, key) for key, val in class_indices.items())
# write dict into json file
# write dict into json file
json_str = json.dumps(inverse_dict, indent=4)
"""
{
    "0": "daisy",
    "1": "dandelion",
    "2": "roses",
    "3": "sunflowers",
    "4": "tulips"
}
"""
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

# load validation set and return a description: Found 364 images belonging to 5 classes.
val_data_gen = validation_image_generator.flow_from_directory(directory=validation_dir, 
                                                            batch_size=batch_size,
                                                            shuffle=False,
                                                            target_size=(im_height, im_width),
                                                            class_mode='categorical') 
total_val = val_data_gen.n

# 通过ImageDataGenerator自动将label转成one-hot coding
sample_training_images, sample_training_labels = next(train_data_gen)
# print(sample_training_labels[0]) # [0. 0. 0. 0. 1.]

# this function will plot images in the form of a grid with 1 row and 5 columns where images are placed
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
# plotImages(sample_training_images[:5])

model = AlexNet_v1(im_height=im_height, im_width=im_width, class_num=5)
##################### Model 2 ######################################
# model = AlexNet_v2(class_num=5)
# model.build((batch_size, 224, 224, 3)) # when using subclass model
####################################################################
model.summary()
# 每层的参数：kernel size * input channel * kernel number + bias
# 第一个卷积层 11 * 11 * 3 * 48 + 48 = 17472

# # 方法一： using keras high level api for training
# # 关于from_logits的超参数，up主说因为在模型定义时用了softmax，所以这里选False
# # 官方说不建议在model过程中使用softmax，而是直接选from_logits=true，这样数据更稳定
# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
#                 loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
#                 metrics=["accuracy"])

# # callbacks可以写成一组函数，如earlystopping，tensorboard等，这里只列出一个，即保存模型。
# # https://www.tensorflow.org/api_docs/python/tf/keras/callbacks
# # h5是keras的参数格式, ckpt是ts的参数格式
# callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath='./save_weights/myAlex.h5',
#                                                 sava_best_only=True,
#                                                 save_weights_only=True,
#                                                 monitor='val_loss')]


# # tf2.1之后，推荐使用fig方法，其中已经含有fit_generator这个处理大规模数据的功能
# history = model.fit(x=train_data_gen,
#                     steps_per_epoch=total_train // batch_size, # 一个epoch有多少batch
#                     epochs=epochs,
#                     validation_data=val_data_gen,
#                     validation_steps=total_val // batch_size,
#                     callbacks=callbacks)


# # plot loss and accuracy image
# history_dict = history.history
# train_loss = history_dict["loss"]
# train_accuracy = history_dict["accuracy"]
# val_loss = history_dict["val_loss"]
# val_accuracy = history_dict["val_accuracy"]

# # figure 1
# plt.figure()
# plt.plot(range(epochs), train_loss, label='train_loss')
# plt.plot(range(epochs), val_loss, label='val_loss')
# plt.legend()
# plt.xlabel('epochs')
# plt.ylabel('loss')

# # figure 2
# plt.figure()
# plt.plot(range(epochs), train_accuracy, label='train_accuracy')
# plt.plot(range(epochs), val_accuracy, label='val_accuracy')
# plt.legend()
# plt.xlabel('epochs')
# plt.ylabel('accuracy')
# plt.show()

# 以下是方法二，比较底层的方法
# using keras low level api for training
loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    train_loss(loss)
    train_accuracy(labels, predictions)
    
@tf.function
def test_step(images, labels):
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)
    
    test_loss(loss)
    test_accuracy(labels, predictions)

best_test_loss = float('inf')
for epoch in range(1, epochs+1):
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()
    for step in range(total_train // batch_size):
        images, labels = next(train_data_gen)
        train_step(images, labels)
    
    for step in range(total_val // batch_size):
        test_images, test_labels = next(val_data_gen)
        test_step(test_images, test_labels)
    
    template = "Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}"
    print(template.format(epoch, train_loss.result(), train_accuracy.result() * 100, test_loss.result(), test_accuracy.result()*100))
    
    if test_loss.result() < best_test_loss:
        model.save_weights("./save_weights/myAlex.ckpt", save_format='tf')
