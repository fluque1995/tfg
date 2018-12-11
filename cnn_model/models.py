from keras import layers, models, regularizers

# CNN model definition
class ModelsDispatcher:

    def basic_model(self, input_shape, num_classes):
        model = models.Sequential()
        model.add(
            layers.Conv1D(
                128,
                5,
                padding='same',
                input_shape=input_shape
            )
        )
        model.add(
            layers.Activation('relu')
        )
        model.add(
            layers.MaxPooling1D()
        )
        model.add(
            layers.Dropout(0.4)
        )
        for i in range(6):
            model.add(
                layers.Conv1D(
                    int(128/(i+1)),
                    5+(2*i),
                    padding='same',
                )
            )
            model.add(
                layers.Activation('relu')
            )
            model.add(
                layers.MaxPooling1D()
            )
            model.add(
                layers.Dropout(0.4)
            )

        model.add(
            layers.GlobalAveragePooling1D()
        )
        model.add(layers.Dense(256))
        model.add(layers.Activation('relu'))
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(128))
        model.add(layers.Activation('relu'))
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(64))
        model.add(layers.Activation('relu'))
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(num_classes))
        model.add(layers.Activation('softmax'))

        return model
