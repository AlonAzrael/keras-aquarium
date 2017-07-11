
from keras.layers import Embedding, Dense, Input, Flatten, Lambda, LSTM, GRU, Bidirectional, TimeDistributed, Layer
from keras.models import Model
from keras import initializers
from keras import backend as K
from keras.utils import to_categorical
import numpy as np



def HierarchicalAttentionRNN(
    max_sents,
    max_sent_length,
    n_classes,
    embeddings=None,
    n_words=None,
    word_dim=50,
    word_hidden_dim=100,
    sent_hidden_dim=100,
):
    """Hierarchical Attention RNN(GRU)

    Two level of lstm network for text Classification, encode sentence by words first, then encode document by sentences.
    Also add attention for both words and sentences.

    Check paper [HIERARCHICAL ATTENTION NETWORKS FOR DOCUMENT CLASSIFICATION](https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf) for more details.

    Parameters
    ----------

    max_sents : number of sentences in a document
    max_sent_length : number of words in a sentence
    n_classes : number of classes
    embeddings :
        use it to initialize word embeddings if applied
    n_words : number of words in vocabulary
    word_dim : dim of word embeddings
    word_hidden_dim : number of word units in rnn
    sent_hidden_dim : number of sentence units in rnn

    Examples
    --------

    import keras
    from keras_aquarium import hatt_rnn
    from scipy.sparse import csr_matrix
    import numpy as np

    # suppose you have a 3D matrix (n_docs * n_sentences_in_doc * n_words_in_sentence), represents documents,
    sequence_docs = np.zeros([n_docs, n_sentences_in_doc, n_words_in_sentence]) # padding zeros
    word_embeddings = load_glove_word_embeddings()
    vocabulary = load_vocabulary()

    model = hatt_rnn.HierarchicalAttentionRNN(
        max_sents,
        max_sent_length,
        n_classes,

        # if use word_embeddings to initialize word embeddings layer
        embeddings=word_embeddings,
        # else
        n_words=len(vocabulary),
        word_dim=50,

        # units in words and sentences gru layer
        word_hidden_dim=100,
        sent_hidden_dim=100,
    )

    model.fit(sequence_docs, labels)
    """

    if embeddings is None:
        # embeddings = np.random.uniform([n_words, word_dim])
        embedding_layer = Embedding(n_words+1,
                                word_dim,
                                input_length=max_sent_length,
                                # mask_zero=True,
                                trainable=True)
    else:
        embedding_layer = Embedding(len(embeddings),
                                len(embeddings[0]),
                                weights=[embeddings],
                                input_length=max_sent_length,
                                mask_zero=True,
                                trainable=True)

    sent_input = Input(shape=(max_sent_length,), dtype='int32')
    embedded_sequences = embedding_layer(sent_input)

    class AttLayer(Layer):
        def __init__(self, hit=None, **kwargs):
            #self.input_spec = [InputSpec(ndim=3)]
            self.init = initializers.glorot_uniform()
            super(AttLayer, self).__init__(**kwargs)
            self.hit = hit

        def build(self, input_shape_li):
            input_shape = input_shape_li[-1]
            assert len(input_shape)==3
            self.W = self.init((input_shape[-1],))
            self.W = K.variable(self.W)
            self._x_input_shape = input_shape
            self.trainable_weights = [self.W]
            super(AttLayer, self).build(input_shape)  # be sure you call this somewhere!

        def call(self, xli, mask=None):
#             eij = K.tanh(K.dot(x, self.W))
            hit, x = xli
            # print "hit.shape:", K.int_shape(hit)

            def get_weights_(x):
                eij = K.dot(x, K.reshape(self.W, [self._x_input_shape[-1], 1]) )
                eij = K.squeeze(eij, axis=-1)
                # print "eij.shape:", K.int_shape(eij)

                ai = K.exp(eij)
                ai_sum = K.sum(ai, axis=1)
                ai_sum = K.reshape(ai_sum, [-1, 1])
                # print "ai_sum.shape:", K.int_shape(ai_sum)
                weights = ai/ai_sum
                # print "weights.shape:", K.int_shape(weights)

                return weights

            weights = get_weights_(x)

            self.output_weights = Lambda(get_weights_, )(x)

            # weighted_input = hit * weights
            weights = K.expand_dims(weights, axis=1)
            weighted_input = K.batch_dot(weights, hit, axes=[2, 1, ])
            weighted_input = K.squeeze(weighted_input, axis=1)

            # weighted_input = K.tf.einsum("ijk,ij->ijk", hit, weights) # batch_dot is equivalent to K.tf.einsum to general method
            # weighted_input = K.sum(weighted_input, axis=1)

            # print "weighted_input.shape:", K.int_shape(weighted_input)

            return weighted_input

        def get_output_shape_for(self, input_shape_li):
            input_shape = input_shape_li[-1]
            return (input_shape[0], input_shape[-1])

        def compute_output_shape(self, input_shape_li):
            return self.get_output_shape_for(input_shape_li)

    def get_weights(args):
        a, b = args
        eij = K.dot(a, K.transpose(b))
        ai = K.exp(eij)
        weights = ai / K.sum(ai, axis=1)
        return weights

    layer_mode = True

    # ======== sent level =========

    sent_hidden = Bidirectional(
        GRU(word_hidden_dim, activation="tanh", return_sequences=True)
    )(embedded_sequences)

    bi_word_hidden_dim = 2 * word_hidden_dim
    sent_hidden_att = TimeDistributed(
        Dense(bi_word_hidden_dim, activation="sigmoid")
    )(sent_hidden)

    if layer_mode:
        word_att_layer = AttLayer()
        sent_encoded = word_att_layer([sent_hidden, sent_hidden_att])
    else:
        words_attention = K.random_uniform_variable(
            [1, bi_word_hidden_dim], low=0, high=1, )
        word_weights = get_weights([sent_hidden_att, words_attention])
        def attend_words(args):
            sent_hidden, sent_hidden_att = args
            weighted_input = sent_hidden * word_weights
            weighted_input = K.sum(weighted_input, axis=1)
            return weighted_input
        sent_encoded = Lambda(attend_words, )([sent_hidden, sent_hidden_att])

    sent_encoder = Model(sent_input, sent_encoded)

    # ======== doc level =========

    sents_input = Input(
        shape=(max_sents, max_sent_length), dtype='int32', )

    sents_encoded = TimeDistributed(sent_encoder)(sents_input)
    doc_hidden = Bidirectional(
        GRU(sent_hidden_dim, activation="tanh", return_sequences=True)
    )(sents_encoded)

    bi_sent_hidden_dim = 2 * sent_hidden_dim
    doc_hidden_att = TimeDistributed(
        Dense(bi_sent_hidden_dim, activation="sigmoid")
    )(doc_hidden)

    if layer_mode:
        sent_att_layer = AttLayer()
        doc_encoded = sent_att_layer([doc_hidden, doc_hidden_att])
    else:
        sents_attention = K.random_uniform_variable(
            [1, bi_sent_hidden_dim], low=0, high=1, )
        sent_weights = get_weights([doc_hidden_att, sents_attention])
        def attend_doc(args):
            doc_hidden, doc_hidden_att = args
            weighted_input = sent_hidden * sent_weights
            weighted_input = K.sum(weighted_input, axis=1)
            return weighted_input
        doc_encoded = Lambda(attend_doc, )([doc_hidden, doc_hidden_att])

    # ======== fully connected =========

    pred = Dense(n_classes, activation='softmax')(doc_encoded)
    model = Model(sents_input, pred)

    model.compile(
        loss='categorical_crossentropy',
        optimizer='nadam',
        metrics=['accuracy'])

    # ======== weights =========

    if layer_mode:
        # pass
        sent_weights_model = Model(sents_input, sent_att_layer.output_weights)
    else:
        word_weights_layer = Lambda(get_weights, )([sent_hidden_att, words_attention])
        print K.int_shape(word_weights_layer)
        word_weights_model = Model(sent_input, word_weights_layer)

        sents_word_weights = TimeDistributed(word_weights_model)(sents_input)
        word_weights_model = Model(sents_input, sents_word_weights)

        sent_weights_layer = Lambda(get_weights, )([doc_hidden_att, sents_attention])
        sent_weights_model = Model(sents_input, sent_weights_layer)

    model._keras_aquarium_params = \
    dict(
        model_type="hatt_rnn",
        # word_weights_model=word_weights_model,
        sent_weights_model=sent_weights_model,
        max_sents=max_sents,
        max_sent_length=max_sent_length,
    )

    return model


def _make_X_Y(docs, labels, max_sents, max_sent_length, n_classes=None):
    X = np.zeros(
        (len(docs), max_sents, max_sent_length), dtype='int32')

    for i, sents in enumerate(docs):
        for j, sent in enumerate(sents):
            if j < max_sents:
                for k, word in enumerate(sent):
                    if k < max_sent_length:
                        X[i,j,k] = word
                    else:
                        break
            else:
                break

    if labels is not None:
        Y = np.zeros([len(labels), n_classes], dtype=np.int32)
        for i, l in enumerate(labels):
            Y[i, l] = 1
        # Y = to_categorical(labels)
        return  X, Y
    else:
        return X


def padding_docs(docs, max_sents, max_sent_length):
    return _make_X_Y(docs, None, max_sents, max_sent_length)


def generate_dataset(
    docs,
    labels,
    max_sents,
    max_sent_length,
    epochs=None,
    batch_size=1024,
    shuffle=True,
    verbose=0,
):

    n_data = len(docs)
    indices = np.arange(n_data)
    n_classes = len(set(labels))

    epoch_i = 0
    while True:
        if shuffle:
            np.random.shuffle(indices)
        if verbose:
            print "run new epoch:", epoch_i

        for i in xrange(0, n_data, batch_size):
            cur_indices = indices[i:i+batch_size]
            doc_batch = [docs[idx] for idx in cur_indices]
            label_batch = [labels[idx] for idx in cur_indices]

            X, Y = _make_X_Y(
                doc_batch, label_batch,
                max_sents, max_sent_length, n_classes)

        yield X, Y


def train_model(model, dataset_generator, steps_per_epoch, **kwargs):
    model.fit_generator(dataset_generator, steps_per_epoch, **kwargs)
    return model


def get_sent_weights(model, docs, ):
    max_sents = model._keras_aquarium_params["max_sents"]
    max_sent_length = model._keras_aquarium_params["max_sent_length"]

    X = _make_X_Y(
        docs, None, max_sents, max_sent_length)

    sent_weights = model._keras_aquarium_params["sent_weights_model"].predict(X)

    return sent_weights
