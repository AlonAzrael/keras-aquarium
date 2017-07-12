
from keras.layers import Input, Dense, Embedding, dot as dot_layer, add as add_layer, Flatten
from keras.models import Model, Sequential
from keras import backend as K
from keras import optimizers, losses, regularizers

from scipy.sparse import csr_matrix


class Dense_tied(Dense):
    """A fully connected layer with tied weights.
    Can be used as a normal keras dense layer
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
    """Softmax AutoEncoder

    Autoencoder using kullback_leibler_divergence as objective function, and softmax as output activation.
    Requiring input matrix row sum to 1.

    Parameters
    ----------

    input_dim : dim of input sample.
    latent_dim : latent dim of latent vector.
    encoder :
        if not None, then will be used as latent_vector = encoder(input_layer).
    decoder :
        if not None, then will be used as generated_input = decoder(latent_vector).
    activation :
        default is "tanh" when use_binary_activation is False, otherwise variant sigmoid.
    loss : default is kullback_leibler_divergence.
    use_tied_layer :
        whether to use tied layer or not,
        used only when encoder and decoder is None.
    use_binary_activation :
        if True, using variant sigmoid 1/(1+exp(alpha*-x)).
    alpha : alpha in variant sigmoid.
    lr : learning rate.

    Examples
    --------

    import keras
    from keras_aquarium import sae
    from scipy.sparse import csr_matrix
    import numpy as np

    # suppose you have a sparse matrix, which represents bag-of-words documents
    bow_docs = csr_matrix([n_docs, n_words])

    model = sae.SoftmaxAutoEncoder(
        input_dim, # dim of input sample
        latent_dim=50, # latent dim of latent vector
        encoder=None, # if not None, then will be used as latent_vector = encoder(input_layer),
        decoder=None, # if not None, then will be used as generated_input = decoder(latent_vector)
        activation=None, # default is "tanh" when use_binary_activation is False, otherwise variant sigmoid
        loss=None, # default is kullback_leibler_divergence
        use_tied_layer=True, # whether to use tied layer or not, used only when encoder and decoder is None
        use_binary_activation=True, # if True, using variant sigmoid 1/(1+exp(alpha*-x))
        alpha=50, # alpha in variant sigmoid
    )

    def generate_dataset(batch_size):
        # memory friendly
        indices = np.arange(len(bow_docs))
        while True:
            np.random.shuffle(indices)
            for i in xrange(0, len(indices), batch_size):
                inds = indices[i:i+batch_size]
                yield bow_docs[inds], bow_docs[inds].toarray()

    batch_size = 32
    model.fit_generator( generate_dataset(batch_size),  len(bow_docs)/batch_size, )
    """

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

    def generate_dataset(batch_size):
        # memory friendly
        indices = np.arange(len(X))
        while True:
            np.random.shuffle(indices)
            for i in xrange(0, len(indices), batch_size):
                inds = indices[i:i+batch_size]
                yield X[inds], X[inds].toarray()

    # Xdense = X.toarray()
    model.fit_generator(generate_dataset(batch_size), len(X)/batch_size, epochs=epochs, verbose=verbose )

    return model


def get_latent_code(model, X):
    encoder = model._keras_aquarium_params["encoder"]
    codes = encoder.predict(X)

    return codes


def evaluate_cluster(y_true, y_pred):
    mi = mutual_info_score(y_true, y_pred)
    ami = adjusted_mutual_info_score(y_true, y_pred)
    comp = completeness_score(y_true, y_pred)
    hom = homogeneity_score(y_true, y_pred)
    vs = v_measure_score(y_true, y_pred)
    return mi, ami, comp, hom, vs
