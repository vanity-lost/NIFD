import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def create_ffn():
    fnn_layers = []

    for units in [32, 32]:
        fnn_layers.append(layers.BatchNormalization())
        fnn_layers.append(layers.Dropout(0.5))
        fnn_layers.append(layers.Dense(units, activation=tf.nn.gelu))

    return keras.Sequential(fnn_layers)

class GraphConvLayer(layers.Layer):
    def __init__(self):
        super(GraphConvLayer, self).__init__()
        self.ffn_prepare = create_ffn()
        self.update_fn = create_ffn()

    def call(self, inputs):
        features, edges, edge_weights = inputs
        neighbour_info = tf.gather(features, edges[1])

        # Prepare the messages of the neighbours.
        messages = self.ffn_prepare(neighbour_info)
        neighbour_messages = messages * tf.expand_dims(edge_weights, -1)

        # mean aggregate neighbors' messages
        aggregated_messages = tf.math.unsorted_segment_mean(
            neighbour_messages, edges[0], num_segments=features.shape[0]
        )

        # feedforward and normalize
        h = tf.concat([features, aggregated_messages], axis=1)
        return tf.nn.l2_normalize(self.update_fn(h), axis=-1)

class GNNNodeClassifier(tf.keras.Model):
    def __init__(self, node_features, edges, num_classes):
        super(GNNNodeClassifier, self).__init__()

        # get the graph info
        self.node_features = node_features
        self.edges = edges
        self.edge_weights = tf.ones(shape=edges.shape[1])
        self.edge_weights = self.edge_weights / tf.math.reduce_sum(self.edge_weights)

        # create the layers
        self.preprocess = create_ffn()
        self.conv1 = GraphConvLayer()
        self.conv2 = GraphConvLayer()
        self.postprocess = create_ffn()
        self.compute_logits = layers.Dense(units=num_classes, name="logits")

    def call(self, input_node_indices):
        # crea the model based on layers
        x = self.preprocess(self.node_features)

        x1 = self.conv1((x, self.edges, self.edge_weights))
        x = x1 + x
        x2 = self.conv2((x, self.edges, self.edge_weights))
        x = x2 + x

        x = self.postprocess(x)
        node_embeddings = tf.gather(x, input_node_indices)
        return self.compute_logits(node_embeddings)

def model_fit(model, x_train, y_train, verbose=0, epoches=300):
    # train the model
    model.compile(
        optimizer=keras.optimizers.Adam(1e-2),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )
    tqdm_callback = tfa.callbacks.TQDMProgressBar(show_epoch_progress=False)
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_acc", patience=50, restore_best_weights=True
    )

    history = model.fit(
        x=x_train,
        y=y_train,
        epochs=epoches,
        batch_size=256,
        validation_split=0.15,
        callbacks=[early_stopping, tqdm_callback],
        verbose=verbose,
    )

    if len(history.history['loss']) < epoches:
        print('Early stopped!')

    return history

def plot_learning_curves(history):
    # plot the learning curves for the model
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(history.history["loss"])
    ax1.plot(history.history["val_loss"])
    ax1.legend(["train", "test"])
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")

    ax2.plot(history.history["acc"])
    ax2.plot(history.history["val_acc"])
    ax2.legend(["train", "test"])
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    plt.show()
