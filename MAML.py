import functools
import numpy
import tensorflow as tf
import toolz
from tqdm import tqdm
from evaluator import RecallEvaluator
from sampler import WarpSampler
import Dataset
import pandas as pd
from tensorflow.contrib.layers.python.layers import regularizers
import os
import gc
import argparse

gc.collect()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def doublewrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if not arguments are provided. All arguments must be optional.
    """
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)

    return decorator

@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    """
    attribute = '_cache_' + function.__name__
    # name = scope or function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            # with tf.variable_scope(name, *args, **kwargs):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


class MAML(object):
    def __init__(self,
                 n_users,
                 n_items,
                 embed_dim=20,
                 batch_size = 10,
                 n_negative=20,
                 imagefeatures=None,
                 textualfeatures=None,
                 margin=1.5,
                 master_learning_rate=0.1,
                 clip_norm=1.0,
                 hidden_layer_dim=128,
                 dropout_rate=0.2,
                 feature_l2_reg=0.1,
                 feature_projection_scaling_factor=0.5,
                 use_rank_weight=True,
                 use_cov_loss=True,
                 cov_loss_weight=0.1
                 ):
        """

        :param n_users: number of users i.e. |U|
        :param n_items: number of items i.e. |V|
        :param embed_dim: embedding size i.e. K (default 20)
        :param features: (optional) the feature vectors of items, shape: (|V|, N_Features).
               Set it to None will disable feature loss(default: None)
        :param margin: hinge loss threshold i.e. z
        :param master_learning_rate: master learning rate for AdaGrad
        :param clip_norm: clip norm threshold (default 1.0)
        :param hidden_layer_dim: the size of feature projector's hidden layer (default: 128)
        :param dropout_rate: the dropout rate between the hidden layer to final feature projection layer
        :param feature_l2_reg: feature loss weight
        :param feature_projection_scaling_factor: scale the feature projection before compute l2 loss. Ideally,
               the scaled feature projection should be mostly within the clip_norm
        :param use_rank_weight: whether to use rank weight
        :param use_cov_loss: use covariance loss to discourage redundancy in the user/item embedding
        """

        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.batch_size = batch_size
        self.n_negative = n_negative
        self.clip_norm = clip_norm
        self.margin = margin
        if imagefeatures is not None:
            self.imagefeatures = tf.constant(imagefeatures, dtype=tf.float32)
        else:
            self.imagefeatures = None
        if textualfeatures is not None:
            self.textualfeatures = tf.constant(textualfeatures, dtype=tf.float32)
        else:
            self.textualfeatures = None
        self.master_learning_rate = master_learning_rate
        self.hidden_layer_dim = hidden_layer_dim
        self.dropout_rate = dropout_rate
        self.feature_l2_reg = feature_l2_reg
        self.feature_projection_scaling_factor = feature_projection_scaling_factor
        self.use_rank_weight = use_rank_weight
        self.use_cov_loss = use_cov_loss
        self.cov_loss_weight = cov_loss_weight
        self.user_positive_items_pairs = tf.placeholder(tf.int32, [self.batch_size, 2])
        self.negative_samples = tf.placeholder(tf.int32, [self.batch_size, self.n_negative])
        self.score_user_ids = tf.placeholder(tf.int32, [None])
        self.max_train_count = tf.placeholder(tf.int32, None)
        self.user_ids = tf.placeholder(tf.int32, [None])
        self.item_ids = tf.placeholder(tf.int32, [None])

        self.user_embeddings
        self.item_embeddings
        self.embedding_loss
        self.feature_loss
        self.loss
        self.optimize


    @define_scope
    def user_embeddings(self):
        return tf.Variable(tf.random_normal([self.n_users, self.embed_dim],
                                            stddev=1 / (self.embed_dim ** 0.5), dtype=tf.float32))
    @define_scope
    def item_embeddings(self):
        return tf.Variable(tf.random_normal([self.n_items, self.embed_dim],
                                            stddev=1 / (self.embed_dim ** 0.5), dtype=tf.float32))

    @define_scope
    def mlp_layer_1(self):
        return tf.layers.dense(inputs=tf.nn.l2_normalize(tf.concat([self.textualfeatures,self.imagefeatures],axis=1),dim=1),
                               units=self.hidden_layer_dim,
                               activation=tf.nn.relu, name="mlp_layer_1")

    @define_scope
    def mlp_layer_2(self):
        dropout = tf.layers.dropout(inputs=self.mlp_layer_1, rate=self.dropout_rate)
        hidden_layer1 = tf.layers.dense(inputs=tf.nn.l2_normalize(dropout,dim=1), units=self.hidden_layer_dim/2 ,activation=tf.nn.relu, name="mlp_layer_2")
        dropout1 = tf.layers.dropout(inputs=hidden_layer1, rate=self.dropout_rate)
        hidden_layer2 = tf.layers.dense(inputs=tf.nn.l2_normalize(dropout1,dim=1), units=self.hidden_layer_dim/2, activation=tf.nn.relu,name="mlp_layer_3")
        dropout2 = tf.layers.dropout(inputs=hidden_layer2, rate=self.dropout_rate)
        return  tf.layers.dense(inputs=dropout2, units=self.embed_dim, name="mlp_layer_4")

    @define_scope
    def feature_projection(self):
        """
        :return: the projection of the feature vectors to the user-item embedding
        """
        # feature loss
        if self.imagefeatures is not None:
            # fully-connected layer
            output = self.mlp_layer_2 * self.feature_projection_scaling_factor

            # projection to the embedding
            return tf.clip_by_norm(output, self.clip_norm, axes=[1], name="feature_projection")

    @define_scope
    def feature_loss(self):
        """
        :return: the l2 loss of the distance between items' their embedding and their feature projection
        """
        loss = tf.constant(0, dtype=tf.float32)
        if self.feature_projection is not None:
            # the distance between feature projection and the item's actual location in the embedding
            feature_distance = tf.reduce_sum(tf.squared_difference(
                self.item_embeddings,
                self.feature_projection), 1)
            # apply regularization weight
            loss += tf.reduce_sum(feature_distance, name="feature_loss") * self.feature_l2_reg
        return loss

    @define_scope
    def covariance_loss(self):
        X = tf.concat((self.item_embeddings, self.user_embeddings), 0)
        n_rows = tf.cast(tf.shape(X)[0], tf.float32)
        X = X - (tf.reduce_mean(X, axis=0))
        cov = tf.matmul(X, X, transpose_a=True) / n_rows
        return tf.reduce_sum(tf.matrix_set_diag(cov, tf.zeros(self.embed_dim, tf.float32))) * self.cov_loss_weight

    @define_scope
    def embedding_loss(self):
        """
        :return: the distance metric loss
        """
        # Let
        # N = batch size,
        # K = embedding size,
        # W = number of negative samples per a user-positive-item pair
        # user embedding (N, K)
        users = tf.nn.embedding_lookup(self.user_embeddings,
                                       self.user_positive_items_pairs[:, 0],
                                       name="users")

        pos_items = tf.nn.embedding_lookup(self.item_embeddings, self.user_positive_items_pairs[:, 1],
                                           name="pos_items")

        # negative item embedding (N, W, K)
        neg_items = tf.nn.embedding_lookup(self.item_embeddings, self.negative_samples, name="neg_items")

        pos_items_f = tf.nn.embedding_lookup(self.feature_projection, self.user_positive_items_pairs[:, 1],
                                             name="pos_items_f")
        neg_items_f = tf.nn.embedding_lookup(self.feature_projection, self.negative_samples, name="neg_items_f")

        input_pos = tf.concat([users,pos_items, pos_items_f], 1,name='input_pos')
        input_neg = tf.reshape(
            tf.concat([tf.tile(tf.expand_dims(users, 1), [1, self.n_negative, 1]),neg_items,neg_items_f], 2),
            [-1, self.embed_dim * 3],name='input_neg')
        input = tf.concat([input_pos, input_neg], 0,name='input')
        with tf.variable_scope("dense"):
            hidden_layer = tf.layers.dense(inputs=tf.nn.l2_normalize(input,dim=1), units=5*self.embed_dim,
                                          kernel_regularizer=regularizers.l2_regularizer(100.0), activation=tf.nn.tanh,
                                         name='hidden_layer')
            dropout = tf.layers.dropout(hidden_layer,0.05)
            hidden_layer1 = tf.layers.dense(inputs=tf.nn.l2_normalize(dropout,dim=1), units=1 * self.embed_dim,
                                            kernel_regularizer=regularizers.l2_regularizer(100.0),
                                            activation=tf.nn.relu,
                                            name='hidden_layer1')
            attention_layer_all = self.embed_dim*tf.nn.softmax(hidden_layer1,dim=-1)

        attention_layer_pos, attention_layer = tf.split(attention_layer_all, [self.batch_size, self.batch_size*self.n_negative], 0)

        #tf.nn.softmax()
        # positive item to user distance (N)
        pos_distances = tf.reduce_sum(
            tf.squared_difference(tf.multiply(attention_layer_pos, users), tf.multiply(attention_layer_pos, pos_items)),
            1, name="pos_distances")
        attention_reshape = tf.transpose(tf.reshape(attention_layer, [-1, self.n_negative, self.embed_dim]), [0, 2, 1])
        # distance to negative items (N x W)
        distance_to_neg_items = tf.reduce_sum(
            tf.squared_difference(tf.multiply(attention_reshape, tf.expand_dims(users, -1)),
                                  tf.multiply(attention_reshape, tf.transpose(neg_items, [0, 2, 1]))), 1,
            name="distance_to_neg_items")
        print('distance_to_neg_items.shape:', distance_to_neg_items.shape)

        # best negative item (among W negative samples) their distance to the user embedding (N)
        closest_negative_item_distances = tf.reduce_min(distance_to_neg_items, 1, name="closest_negative_distances")
        print('closest_negtive_item_distances:',closest_negative_item_distances.shape)
        # compute hinge loss (N)
        loss_per_pair = tf.maximum(pos_distances - closest_negative_item_distances + self.margin, 0,
                                   name="pair_loss")

        if self.use_rank_weight:
            # indicator matrix for impostors (N x W)
            impostors = (tf.expand_dims(pos_distances, -1) - distance_to_neg_items + self.margin) > 0
            # approximate the rank of positive item by (number of impostor / W per user-positive pair)
            rank = tf.reduce_mean(tf.cast(impostors, dtype=tf.float32), 1, name="rank_weight") * self.n_items
            # apply rank weight
            loss_per_pair *= tf.log(rank + 1)
        # the embedding loss
        loss = tf.reduce_sum(loss_per_pair, name="loss")
        return loss

    @define_scope
    def loss(self):
        """
        :return: the total loss = embedding loss + feature loss
        """
        loss = self.embedding_loss + self.feature_loss
        if self.use_cov_loss:
            loss += self.covariance_loss
        return loss

    @define_scope
    def clip_by_norm_op(self):
        return [tf.assign(self.user_embeddings, tf.clip_by_norm(self.user_embeddings, self.clip_norm, axes=[1])),
                tf.assign(self.item_embeddings, tf.clip_by_norm(self.item_embeddings, self.clip_norm, axes=[1]))]

    @define_scope
    def optimize(self):
        # have two separate learning rates. The first one for user/item embedding is un-normalized.
        # The second one for feature projector NN is normalized by the number of items.
        gds = []
        gds.append(tf.train
                   .AdamOptimizer(self.master_learning_rate)
                   .minimize(self.loss, var_list=[self.user_embeddings, self.item_embeddings])) #AdagradOptimizer
        with tf.control_dependencies(gds):
            return gds + [self.clip_by_norm_op]

    @define_scope
    def item_scores(self):
        # (N_USER_IDS, 1, K)
        user = tf.expand_dims(tf.nn.embedding_lookup(self.user_embeddings, self.score_user_ids), 1,name='user_test')
        # (1, N_ITEM, K)
        item = tf.tile(tf.expand_dims(self.item_embeddings, 0), [tf.shape(user)[0], 1, 1],name='item_test')
        feature = tf.tile(tf.expand_dims(self.feature_projection, 0), [tf.shape(user)[0], 1, 1],name='feature_test')
        input = tf.concat(
            [tf.reshape(tf.tile(user, [1, tf.shape(item)[1], 1]), [-1, self.embed_dim]),
             tf.reshape(item, [-1, self.embed_dim]),tf.reshape(feature, [-1, self.embed_dim])], 1,name='input_test')

        with tf.variable_scope('dense'):
            hidden_layer = tf.layers.dense(inputs=tf.nn.l2_normalize(input, dim=1), units=5 * self.embed_dim,
                                           trainable=False,
                                           kernel_regularizer=regularizers.l2_regularizer(100.0), activation=tf.nn.tanh,
                                           name='hidden_layer', reuse=True)

            hidden_layer1 = tf.layers.dense(inputs=tf.nn.l2_normalize(hidden_layer, dim=1), units=1 * self.embed_dim,
                                             trainable=False,
                                             kernel_regularizer=regularizers.l2_regularizer(100.0),
                                             activation=tf.nn.relu,
                                             name='hidden_layer1', reuse=True)
            attention_layer_score = self.embed_dim * tf.nn.softmax(hidden_layer1, dim=-1)

        attention_reshape = tf.reshape(attention_layer_score, [-1, tf.shape(item)[1], self.embed_dim],name='attention_test')
        scores = -tf.reduce_sum(
            tf.squared_difference(tf.multiply(attention_reshape, user), tf.multiply(attention_reshape, item)), 2,
            name="scores")
        top_n = tf.nn.top_k(scores, 10 + self.max_train_count,name='top_n')
        return top_n

def load_rating_file_as_list(filename):
    ratingList = []
    with open(filename, "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split("\t")
            user, item = int(arr[0]), int(arr[1])
            ratingList.append([user, item])
            line = f.readline()
    return ratingList
def optimize(model, sampler, train, test, args):
    """
    Optimize the model. TODO: implement early-stopping
    :param model: model to optimize
    :param sampler: mini-batch sampler
    :param train: train user-item matrix
    :param valid: validation user-item matrix
    :return: None
    """
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    #sess.graph.finalize()
    if model.feature_projection is not None:
        # initialize item embedding with feature projection
        sess.run(tf.assign(model.item_embeddings, model.feature_projection))
    # all test users to calculate recall validation
    test_users = numpy.asarray(list(set(test.nonzero()[0])),dtype=numpy.int32)
    testresult = RecallEvaluator(model, train, test)
    epoch = 0
    tempbest = 0
    while True:
        print('\nepochs:{}'.format(epoch))
        epoch += 1
        # TODO: early stopping based on validation recall
        # train model
        losses = []
        feature_losses = []
        # run n mini-batches
        for _ in tqdm(range(args.eva_batches), desc="Optimizing..."):
	    user_pos, neg = sampler.next_batch()
            _, loss ,feature_loss = sess.run((model.optimize, model.loss ,model.feature_loss),
                               {model.user_positive_items_pairs: user_pos,
                                model.negative_samples: neg})
            feature_losses.append(feature_loss)
        #tf.train.Saver.save(sess,'my-model')
        # compute recall,ndcg,hr,pr on test set
        test_recalls, test_ndcg, test_hr, test_pr = [], [], [], []
        for user_chunk in toolz.partition_all(100, test_users):
            recalls, ndcgs, hit_ratios, precisions = testresult.eval(sess, user_chunk)
            test_recalls.extend(recalls)
            test_ndcg.extend(ndcgs)
            test_hr.extend(hit_ratios)
            test_pr.extend(precisions)
        print("\nresult on test set: ndcg:{},recall:{},hr:{},pr:{}".format(sum(test_ndcg)/float(len(test_ndcg)),sum(test_recalls)/float(len(test_recalls)),sum(test_hr)/float(len(test_hr)),sum(test_pr)/float(len(test_pr))))

def parse_args():
    parser = argparse.ArgumentParser(description='Run MAML.')
    parser.add_argument('--dataset', nargs='?',default='Office', help='Choose a dataset.')
    parser.add_argument('--eva_batches', type=int,default=100, help = 'evaluation every n bathes.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
    parser.add_argument('--num_neg', type=int, default=4, help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--hidden_layer_dim', type=int, default=256, help='Hidden layer dim.')
    parser.add_argument('--margin', type=float, default=1.0, help='margin.' )
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout.' )
    parser.add_argument('--feature_l2_reg', type=float, default=1.0, help='feature_l2_reg')
    args = parser.parse_args();
    return args

if __name__ == '__main__':
    # get user-item matrix
    # make feature as dense matrix
    args = parse_args()
    Filename = args.dataset
    Filepath = 'Data/'+ Filename
    dataset = Dataset.Dataset(Filepath)
    train, test = dataset.trainMatrix, dataset.testRatings
    textualfeatures, imagefeatures = dataset.textualfeatures, dataset.imagefeatures
    # print(type(features))

    n_users, n_items = max(train.shape[0],test.shape[0]),max(train.shape[1],test.shape[1])
    print(n_users,n_items)
    # create warp sampler
    sampler = WarpSampler(train, batch_size=args.batch_size, n_negative=args.num_neg, check_negative=True)

    # WITHOUT features
    # Train a user-item joint embedding, where the items a user likes will be pulled closer to this users.
    # Once the embedding is trained, the recommendations are made by finding the k-Nearest-Neighbor to each user.

    model = MAML(n_users,
                n_items,
                # enable feature projection
                imagefeatures=imagefeatures,
                textualfeatures=textualfeatures,
                embed_dim=64,
                batch_size=args.batch_size,
                # N_negatvie
                n_negative=args.num_neg,
                margin=args.margin,
                clip_norm=1.0,
                master_learning_rate=0.001,
                # the size of the hidden layer in the feature projector NN
                hidden_layer_dim=args.hidden_layer_dim,
                # dropout rate between hidden layer and output layer in the feature projector NN
                dropout_rate=args.dropout,
                # scale the output of the NN so that the magnitude of the NN output is closer to the item embedding
                feature_projection_scaling_factor=1.0,
                # the penalty to the distance between projection and item's actual location in the embedding
                # tune this to adjust how much the embedding should be biased towards the item features.
                feature_l2_reg=args.feature_l2_reg,
                # whether to enable rank weight. If True, the loss will be scaled by the estimated
                # log-rank of the positive items. If False, no weight will be applied.
                # This is particularly useful to speed up the training for large item set.
                # Weston, Jason, Samy Bengio, and Nicolas Usunier.
                # "Wsabie: Scaling up to large vocabulary image annotation." IJCAI. Vol. 11. 2011.
                use_rank_weight=True,
                # whether to enable covariance regularization to encourage efficient use of the vector space.
                # More useful when the size of embedding is smaller (e.g. < 20 ).
                use_cov_loss=True,
                # weight of the cov_loss
                cov_loss_weight=5.0
                )
    optimize(model, sampler, train,test,args)
