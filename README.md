## keras-aquarium is a small collection of models powered by keras  

In this repository, following models are included:

- __Deep Matrix Factorization(Recommendation System)__ , by introducing deeper encoding of both user vectors and item vectors, it outperform than simple Matrix Factorization, the model is inspried by paper [COLLABORATIVE DEEP EMBEDDING VIA DUAL NETWORKS ](https://openreview.net/pdf?id=r1w7Jdqxl).

- __Deep Document Modeling__ , many papers talking about using traditional autoencoder for document modeling, but according to my experiments, it did not work well. Replacing objective function with kullback\_leibler\_divergence solve the problem, it can be an alternative for LDA.

- __Deep Text Classification__ , implementation follows the paper [HIERARCHICAL ATTENTION NETWORKS FOR DOCUMENT CLASSIFICATION](https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf), and can be used for important sentences extraction.


------------------------------------------------------  


### Tutorials for Deep Matrix Factorization  

```python
import keras
from keras_aquarium import dmf
from scipy.sparse import csr_matrix, coo_matrix

# suppose you have a sparse matrix as a user item rating matrix
rating_matrix = coo_matrix( shape=[n_user, n_item] )
users = rating_matrix.row
items = rating_matrix.col
ratings = rating_matrix.data

# you also have a sparse feature matrix for item, such as location of item, price of item, etc.
item_features = csr_matrix( shape=[n_item, n_item_feature] )
# and sadly, you dont have any feature for user
user_features = None

# then you want to apply a Matrix Factorization model to predict ratings
model = dmf.DualMatrixFactorization(
    # we choose user as row, item as col
    n_row=n_user, n_col=n_item, # specified matrix shape
    # or we can choose item as row, user as col, n_row=n_item, n_col=n_user, just transpose the matrix

    n_row_feature=None, # we dont have user feature
    n_col_feature=n_item_feature, # we do have item feature
    row_dim=30, col_dim=20, # specifed embedding dim, each user and item is then first embedded as a vector
    row_feature_dim=None, # no user feature
    col_feature_dim=None, # specifed item feature embedding dim, each item feature will be encoded as a dense vector

    # row_layers and col_layers are for encoding layers for user and item,
    # they work separately, so they can have different number of hidden layers
    # just make sure the finla hidden layer are the same dim
    row_layers=[(50, "relu"), (30, keras.losses.tanh)], # use two hidden layers, first one is a dense layer with 50 unit and relu activation, second one is a dense layer with 30 unit and tanh activation
    col_layers=[ (lambda x: Dense(30, regularizer=l2(0.001))(x) ), ], # can use a callable to create a hidden layer
)

# use it as a keras model
model.compile(loss="mse", optimizer="adam", )

inputs = [users] + [items, item_features[items], ] # note that there is no user_features
model.fit(inputs, ratings, )
```


### Tutorials for Deep Topic Modeling

```python
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

# after training model, we can run kmeans to get topic distribution
from sklearn.cluster import KMeans
km = KMeans(n_clsuters=20)
y_pred = km.fit_predict(sae.get_latent_code(bow_docs))

# scoring our topic by comparing topic and true document labels
from sklearn.metrics import homogeneity_completeness_v_measure
homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(y_true, y_pred)
```


### Tutorials for Deep Text Classification

```python
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
```

------------------------------------------------------  

## More about how and why above models work

#### Deep Matrix Factorization
When it comes to deep matrix Factorization, the model described in this [page](http://blog.richardweiss.org/2016/09/25/movie-embeddings.html) is easier to think about.
However, as the page points out, a major disadvantages of such architecture is its efficiency.
To predict a rating, you must run through network, which is quite computational expensive.
Instead, dual encoding network, can first transform each row and col into latent vector, and run a simple vector dot operation to predict rating.
<br/>
<br/>

#### Deep Document Modeling
Traditional autoencoder using mean squared error as objective function.

However it didn't work for bag-of-words, as bag-of-words is very sparse.
Consider we have a 20,000 words vocabulary, that means we have 20,000 output units of our autoencoder.
And the average unique word numbers of a document is 200, that means, only 200/20,000 units will be nonzero, which is 1%.

So, what's the problem? In the mnist case, we have 20*20 binary image, now suppose only 1% pixels are nonzero, which is 4 pixels in a 20*20 image!
When the nonzero becomes smaller, the harder autoencoder can learn a good latent representation, since mean squared error treat every pixels the same, but we only care about the nonzero pixels!

Thus, we need a objective function which only focus on error of nonzero units, but also help keeping zero units remain zero.
In keras, we have categorical_crossentropy and kullback_leibler_divergence (they are pretty much the same) that focus only nonzero units, and we can use softmax activation to keep zero units remain zero.

After training autoencoder, we extract latent vector of each input, and apply clustering(such as KMeans) and approximate nearest neighbor search algorithms, to get a good topic modeling result, and document retrieval.

If use binary activation, we can significantly improving efficiency of clustering and approximate nearest neighbor search.
<br/>
<br/>

#### Deep Text Classification
Two level of lstm network for text Classification, encode sentence by words first, then encode document by sentences.
Also add attention for both words and sentences.

Check paper [HIERARCHICAL ATTENTION NETWORKS FOR DOCUMENT CLASSIFICATION](https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf) for more details.
Another [very good blog](https://explosion.ai/blog/deep-learning-formula-nlp) about this model.

As Google said, attention is all we need.
