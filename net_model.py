import tensorflow as tf
from keras.layers import (
    Conv2D,
    Dropout,
    MaxPooling2D,
    AveragePooling2D,
    Flatten,
    Dense,
    BatchNormalization,
    Activation,
)


def net_model(base_filtros, train_images,regularizers, w_regularizer, n_clases):
    """Modelo de la red"""
    model = tf.keras.Sequential()

    # Conv 1
    model.add(
        Conv2D(
            filters=base_filtros,
            kernel_size=(3, 3),
            padding="same",
            kernel_regularizer=regularizers.l2(w_regularizer),
            input_shape=train_images.shape[1:],
        )
    )
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    # - - - - - - - - - - - - - - - - - - - - -- - -
    # Conv 3
    model.add(
        Conv2D(
            filters=4 * base_filtros,
            kernel_size=(3, 3),
            padding="same",
            kernel_regularizer=regularizers.l2(w_regularizer),
        )
    )
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    # - - - - - - - - - - - - - - - - - - - - -- - -
    # Conv 6
    model.add(
        Conv2D(
            filters=16 * base_filtros,
            kernel_size=(3, 3),
            padding="same",
            kernel_regularizer=regularizers.l2(w_regularizer),
        )
    )
    model.add(Activation("relu"))
    model.add(AveragePooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.2))

    # - - - - - - - - - - - - - - - - - - - - -- - -
    ## Clasificación - Flatten
    model.add(Flatten())
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(n_clases, activation="softmax"))
    return model
