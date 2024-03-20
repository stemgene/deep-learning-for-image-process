from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, Flatten, Dense, AvgPool2D, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet import ResNet50
import numpy as np

def load_train(path):
    # train_datagen = ImageDataGenerator(validation_split=0.25, rescale=1./255, horizontal_flip=True, vertical_flip=True)
    # validation_datagen = ImageDataGenerator(validation_split=0.25, rescale=1.0 / 255)
    datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rescale=1.0 / 255)

    train_datagen_flow = datagen.flow_from_directory(
        path,
        target_size=(150, 150),
        batch_size=16,
        class_mode='sparse',
        subset='training',
        seed=12345)

    # val_datagen_flow = validation_datagen.flow_from_directory(
    #     '/datasets/fruits_small/',
    #     target_size=(150, 150),
    #     batch_size=16,
    #     class_mode='sparse',
    #     subset='validation',
    #     seed=12345)

    return train_datagen_flow

def create_model(input_shape=(150, 150, 3)):
    optimizer = Adam(lr=0.0001)
    backbone = ResNet50(input_shape=input_shape, weights='imagenet', include_top=False)
    model = Sequential()
    model.add(backbone)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(units=12, activation='softmax'))    

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['acc'],
    )

    return model

def train_model(
    model,
    train_data,
    test_data,
    batch_size=None,
    epochs=4,
    steps_per_epoch=None,
    validation_steps=None
):
    #features_train, target_train = train_data
    #features_test, target_test = test_data
    if steps_per_epoch is None:
        steps_per_epoch = len(train_data)
    if validation_steps is None:
        validation_steps = len(test_data)
    model.fit(
        train_data,
        validation_data=test_data,
        batch_size=batch_size,
        epochs=epochs,
        steps_per_epoch = steps_per_epoch,
        validation_steps = validation_steps,
        verbose=2,
        #shuffle=True,
    )
    return model

