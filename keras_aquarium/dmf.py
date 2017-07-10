
from keras.layers import Input, Dense, Embedding, dot as dot_layer, add as add_layer, Flatten, concatenate
from keras.models import Model, Sequential
from keras import backend as K
from keras import optimizers
from keras import losses

import numpy as np



def DeepMatrixFactorization(
    n_row,
    n_col,
    n_row_feature=None,
    n_col_feature=None,
    row_dim=50,
    col_dim=50,
    row_feature_dim=50,
    col_feature_dim=50,
    row_layers=[(50, "relu")],
    col_layers=[(50, "relu")],
    # output_mode="single",
    model_name=None,
):

    def make_row_layers(n_row, n_row_feature, row_feature_dim):
        row_input = Input(shape=(1,), dtype="int32")
        row_embd = Embedding(
            input_dim=n_row,
            input_length=1,
            output_dim=row_dim,
        )
        row_embd = Flatten()(row_embd(row_input))

        if n_row_feature is not None:
            row_feature_input = Input(shape=(n_row_feature, ), sparse=True)
            row_feature_embd = Dense(
                row_feature_dim,
                activation=None,
            )(row_feature_input)
            row_hidden = concatenate([row_embd, row_feature_embd], )
        else:
            row_feature_input = None
            row_hidden = row_embd

        if row_feature_input is None:
            inputs = [row_input]
        else:
            inputs = [row_input, row_feature_input]

        # print "row_hidden.shape:", K.int_shape(row_hidden)

        return inputs, row_hidden

    [row_inputs, row_hidden] = make_row_layers(
        n_row, n_row_feature, row_feature_dim)
    [col_inputs, col_hidden] = make_row_layers(
        n_col, n_col_feature, col_feature_dim)

    row_hiddens = []
    col_hiddens = []

    def map_layers(layers, hidden):
        hiddens = []

        for l in layers:
            if callable(l):
                hidden = l(hidden)
                print "callable"
            else:
                (dim, act) = l
                print K.int_shape(hidden)
                hidden = Dense(dim, activation=act)(hidden)

            hiddens.append(hidden)

        return hiddens

    row_hiddens = map_layers(row_layers, row_hidden)
    col_hiddens = map_layers(col_layers, col_hidden)

    def zip_row_col(row_hiddens, col_hiddens):
        outputs = []

        for row, col in zip(row_hiddens, col_hiddens):
            output = dot_layer([row, col], axes=-1)
            outputs.append(output)

        return outputs

    outputs = zip_row_col(row_hiddens, col_hiddens)
    # TODO: add multilevel mode
    # if output_mode == "single":
    #     pred = outputs[-1]
    # else:
    #     pred = add_layer(outputs)
    pred = outputs[-1]

    model = Model(
        inputs=row_inputs + col_inputs,
        outputs=pred,
        name=model_name,
    )

    model.compile(
        optimizer=optimizers.Nadam(),
        loss=losses.mean_squared_error,
        metrics=['accuracy']
    )

    model._keras_aquarium_params = \
    dict(
        model_type="dmf",
        outputs=outputs,
        row_hiddens=row_hiddens,
        col_hiddens=col_hiddens,
        row_encoder=Model(inputs=row_inputs, outputs=row_hiddens[-1]),
        col_encoder=Model(inputs=col_inputs, outputs=col_hiddens[-1]),
    )

    return model


def generate_dataset(
    sparse_matrix,
    row_features=None,
    col_features=None,
    epochs=None,
    batch_size=1024,
    shuffle=True,
    verbose=0,
):
    if type(sparse_matrix) == list:
        [row, col, data] = sparse_matrix
        interactions = np.asarray([row, col, data]).T
    else:
        sparse_matrix = sparse_matrix.tocoo()
        interactions = np.asarray([getattr(sparse_matrix, n) for n in ["row", "col", "data"]]).T

    n_data = interactions.shape[0]
    indices = np.arange(n_data)

    epoch_i = 0
    while True:
        if shuffle:
            np.random.shuffle(indices)
        if verbose:
            print "run new epoch:", epoch_i

        for i in xrange(0, n_data, batch_size):

            batch = interactions[ indices[i:i+batch_size] ]
            [row_batch, col_batch, data_batch] = batch.T
            row_batch = row_batch.astype(np.int32)
            col_batch = col_batch.astype(np.int32)

            if row_features is not None:
                row_features_batch = row_features[row_batch]
                row_inputs = [row_batch, row_features_batch]
            else:
                row_features_batch = None
                row_inputs = [row_batch]

            if col_features is not None:
                col_features_batch = col_features[col_batch]
                col_inputs = [col_batch, col_features_batch]
            else:
                col_features_batch = None
                col_inputs = [col_batch]

            yield (row_inputs+col_inputs, data_batch)

        epoch_i += 1
        if epochs is not None and epoch_i >= epochs:
            break


def train_model(model, dataset_generator, steps_per_epoch, **kwargs):
    model.fit_generator(dataset_generator, steps_per_epoch, **kwargs)
    return model


def get_hiddens(model, indices, features=None, name="row"):
    encoder = model._keras_aquarium_params[name+"_encoder"]
    if features is None:
        inputs = [indices]
    else:
        inputs = [indices, features[indices]]

    hiddens = encoder.predict(inputs)

    return hiddens


def get_row_hiddens(*args, **kwargs):
    return get_hiddens(*args, name="row", **kwargs)


def get_col_hiddens(*args, **kwargs):
    return get_hiddens(*args, name="col", **kwargs)
