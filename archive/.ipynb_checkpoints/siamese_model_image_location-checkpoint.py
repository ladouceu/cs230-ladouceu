"""Siamese Network with object Image and Location.

This module contains the code for the siamese network that uses the object image
and location as features.

"""

import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
from pathlib import Path
from tensorflow.keras import applications
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import Model
from tensorflow.keras.applications import resnet
from tensorflow.keras.models import load_model


def visualize(anchor, positive, negative):
    """Visualize a few triplets from the supplied batches."""

    def show(ax, image):
        ax.imshow(image)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    fig = plt.figure(figsize=(9, 9))

    axs = fig.subplots(3, 3)
    for i in range(3):
        show(axs[i, 0], anchor[i])
        show(axs[i, 1], positive[i])
        show(axs[i, 2], negative[i])

def get_siamese_network(target_shape):      
    base_cnn = resnet.ResNet50(
        weights="imagenet", input_shape=target_shape + (3,), include_top=False
    )
    location_input = layers.Input(shape=(4,))

    flatten = layers.Flatten()(base_cnn.output)
    dense1 = layers.Dense(256, activation="relu")(flatten) # TODO: made it simpler by going from 512 to 256
    dense1 = layers.BatchNormalization()(dense1)
    concat = layers.Concatenate()([dense1, location_input])
    dense2 = layers.Dense(128, activation="relu")(concat) # TODO: made it simpler by going from 256 to 128
    dense2 = layers.BatchNormalization()(dense2)
    output = layers.Dense(128)(dense2)# TODO: made it simpler by going from 256 to 128

    embedding = Model([base_cnn.input,location_input], output, name="Embedding")

    trainable = False
    for layer in base_cnn.layers:
        if layer.name == "conv5_block3_out": #TODO: I change from block1 to block3 to reduce number of trainable params
            trainable = True
        layer.trainable = trainable


    class DistanceLayer(layers.Layer):
        """
        This layer is responsible for computing the distance between the anchor
        embedding and the positive embedding, and the anchor embedding and the
        negative embedding.
        """

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def call(self, anchor, positive, negative):
            ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
            an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
            return (ap_distance, an_distance)


    anchor_input_image = layers.Input(name="anchor_image", shape=target_shape + (3,))
    positive_input_image = layers.Input(name="positive_image", shape=target_shape + (3,))
    negative_input_image = layers.Input(name="negative_image", shape=target_shape + (3,))

    anchor_input_location = layers.Input(name="anchor_location",shape=(4,))
    positive_input_location = layers.Input(name="positive_location",shape=(4,))
    negative_input_location = layers.Input(name="negative_location",shape=(4,))



    distances = DistanceLayer()(
        embedding((resnet.preprocess_input(anchor_input_image), anchor_input_location)),
        embedding((resnet.preprocess_input(positive_input_image), positive_input_location)),
        embedding((resnet.preprocess_input(negative_input_image), negative_input_location)),
    )

    siamese_network = Model(
        inputs=[(anchor_input_image,   anchor_input_location),
                (positive_input_image, positive_input_location), 
                (negative_input_image, negative_input_location)],
        outputs=distances
    )
    
    return siamese_network



class SiameseModel(Model):
    """The Siamese Network model with a custom training and testing loops.

    Computes the triplet loss using the three embeddings produced by the
    Siamese Network.

    The triplet loss is defined as:
       L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)
    """

    def __init__(self, siamese_network, margin=0.5):
        super(SiameseModel, self).__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        # GradientTape is a context manager that records every operation that
        # you do inside. We are using it here to compute the loss so we can get
        # the gradients and apply them using the optimizer specified in
        # `compile()`.
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        # Let's update and return the training loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        # Let's update and return the loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        # The output of the network is a tuple containing the distances
        # between the anchor and the positive example, and the anchor and
        # the negative example.
        ap_distance, an_distance = self.siamese_network(data)

        # Computing the Triplet Loss by subtracting both distances and
        # making sure we don't get a negative value.
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker]
    
    

def get_siamese_model(target_shape, load_model_path = None):
    siamese_network = get_siamese_network(target_shape)
    if load_model_path is not None:
        siamese_network = load_model(load_model_path)
    siamese_model = SiameseModel(siamese_network)
    siamese_model.compile(optimizer=optimizers.Adam(0.0001))
    return siamese_model
    # siamese_model.fit(train_dataset, epochs=10, validation_data=val_dataset, batch_size = batch_size)
