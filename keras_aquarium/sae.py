
from keras.layers import Input, Dense, Embedding, dot as dot_layer, add as add_layer, Flatten
from keras.models import Model, Sequential
from keras import backend as K
from keras import optimizers, losses, regularizers

from scipy.sparse import csr_matrix


class Dense_tied(Dense):
    """
    A fully connected layer with tied weights.
    """
    def __init__(self, units,
                 activation=None, use_bias=True,
                 bias_initializer='zeros',
                 kernel_regularizer=None, bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None, bias_constraint=None,
                 tied_to=None, **kwargs):
        self.tied_to = tied_to

        super(Dense_tied, self).__init__(units=units,
                 activation=activation, use_bias=use_bias,
                 bias_initializer=bias_initializer,
                 kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                 activity_regularizer=activity_regularizer,
                 kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
                 **kwargs)

    def build(self, input_shape):
        super(Dense_tied, self).build(input_shape)  # be sure you call this somewhere!
        if self.kernel in self.trainable_weights:
            self.trainable_weights.remove(self.kernel)

    def call(self, x, mask=None):
        # Use tied weights, with tied_to layer
        self.kernel = K.transpose(self.tied_to.kernel)
        output = K.dot(x, self.kernel)
        if self.use_bias:
            output += self.bias
        return self.activation(output)

def SoftmaxAutoEncoder(
    input_dim,
    latent_dim=50,
    encoder=None,
    decoder=None,
    activation=None,
    loss=None,
    sparse=True, use_tied_layer=True, use_binary_activation=True, alpha=50,
    lr=0.001,
):
    input_layer = Input(shape=[input_dim,], sparse=sparse)

    if encoder is not None:
        hidden = encoder(input_layer)
    else:
        hidden = input_layer

    if activation is None:

        if use_binary_activation:
            def binary_activation(x, ):
                """
                embarrsingly using variant sigmoid sgm(x) = 1/(1 + exp(alpha*-x)), turn to sigmoid when alpha = 1,
                faster convergence,
                """
                x = -1*alpha*x
                x = K.clip(x, -1e16, 80)
                alive = 1 / (1+K.exp(x))
                return alive
            activation = binary_activation
        else:
            activation = activations.tanh

    encoder_ = Dense(latent_dim, activation=activation, kernel_initializer="glorot_normal",)
    code = encoder_(hidden)
    # code = Lambda(activation, )(code) # activation

    if decoder is not None:
        hidden_g = decoder(code)
    else:
        hidden_g = code

    if use_tied_layer:
        decoder_ = Dense_tied(input_dim,
            activation="softmax",
            tied_to=encoder_,
            kernel_regularizer=regularizers.l2(0.00001),
            bias_regularizer=regularizers.l2(0.00001),)
    else:
        decoder_ = Dense(input_dim, activation="softmax", )

    res_input = decoder_(hidden_g)

    model = Model(inputs=input_layer, outputs=res_input, )
    if loss is None:
        loss = losses.kullback_leibler_divergence
    model.compile(
        loss=loss,
        optimizer=optimizers.Nadam(lr=lr))

    encoder = Model(inputs=input_layer, outputs=code, )
    model._keras_aquarium_params = \
    dict(encoder=encoder, )

    return model


def train_model(model, X, epochs=5, verbose=2, batch_size=32):
    assert type(X) == csr_matrix

    # TODO: a solution use sparse matrix as output is required, check out Keras code for detail, or use fit_generator
    Xdense = X.toarray()

    model.fit(X, Xdense, epochs=epochs, verbose=verbose, batch_size=batch_size, )

    return model


def get_latent_code(model, X):
    encoder = model._keras_aquarium_params["encoder"]
    codes = encoder.predict(X)

    return codes
