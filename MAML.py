import functools
import numpy
import tensorflow as tf
import logging

import toolz
from tensorflow.contrib.layers.python.layers import regularizers
from evaluator import RecallEvaluator
from sampler import WarpSampler
import Dataset
import os
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


class CML(object):
    def __init__(self,
                 n_users,
                 n_items,
                 embed_dim=20,
                 batch_size = 10,
                 n_negative=20,
                 eval_num=30,
                 t_features=None,
                 v_features=None,
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
        self.eval_num = eval_num
        self.clip_norm = clip_norm
        self.margin = margin
        self.t_features = t_features
        self.v_features = v_features
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
        #all_train_data
        self.user_ids = tf.placeholder(tf.int32,[None])
        self.item_ids = tf.placeholder(tf.int32,[None])

        self.user_embeddings
        self.item_embeddings
        self.embedding_loss
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

        #features
        pos_t_f = tf.nn.embedding_lookup(self.t_features, self.user_positive_items_pairs[:, 1],
                                           name="pos_t_f")
        neg_t_f = tf.nn.embedding_lookup(self.t_features, self.negative_samples, name="neg_t_f")

        pos_v_f = tf.nn.embedding_lookup(self.v_features, self.user_positive_items_pairs[:, 1],
                                         name="pos_v_f")
        neg_v_f = tf.nn.embedding_lookup(self.v_features, self.negative_samples, name="neg_v_f")

        pos_items_concat = tf.concat([pos_items,pos_t_f,pos_v_f],1)
        neg_items_concat = tf.concat([neg_items,neg_t_f,neg_v_f],2)

        input_concat = tf.concat([pos_items_concat,tf.reshape(neg_items_concat,[self.batch_size*self.n_negative,-1])],0)
        with tf.variable_scope("concat_dense"):
            hidden_layer = tf.layers.dense(inputs=input_concat,units=8*self.embed_dim,activation=tf.nn.relu,kernel_regularizer=regularizers.l2_regularizer(0.01),name='hidden_layer')
            dropout = tf.layers.dropout(inputs=hidden_layer, rate=self.dropout_rate)
            hidden_layer1 = tf.layers.dense(inputs=dropout,units=4*self.embed_dim,activation=tf.nn.relu,kernel_regularizer=regularizers.l2_regularizer(0.01),name='hidden_layer1')
            dropout1 = tf.layers.dropout(inputs=hidden_layer1, rate=self.dropout_rate)
            hidden_layer2 = tf.layers.dense(inputs=dropout1,units=2*self.embed_dim,activation=tf.nn.relu,kernel_regularizer=regularizers.l2_regularizer(0.01),name='hidden_layer2')
            dropout2 = tf.layers.dropout(inputs=hidden_layer2, rate=self.dropout_rate)
            output = tf.layers.dense(inputs=dropout2,units=self.embed_dim,kernel_regularizer=regularizers.l2_regularizer(0.01),name='output')
        pos_items_f,neg_items_f = tf.split(output,[self.batch_size,self.batch_size*self.n_negative],0)
        neg_items_f = tf.reshape(neg_items,[self.batch_size,self.n_negative,self.embed_dim])

        input_pos = tf.concat([users, pos_items_f], 1)
        input_neg = tf.reshape(
            tf.concat([tf.tile(tf.expand_dims(users, 1), [1, self.n_negative, 1]), neg_items_f], 2),
            [-1, self.embed_dim * 2])
        input = tf.concat([input_pos, input_neg], 0)
        logging.log(logging.INFO, 'self.input:'+str(input.shape))

        with tf.variable_scope("dense"):
            hidden_layer = tf.layers.dense(inputs=input, units=1*self.embed_dim,
                                          kernel_regularizer=regularizers.l2_regularizer(0.001), activation=tf.nn.tanh,
                                         name='hidden_layer')
            hidden_layer1 = tf.layers.dense(inputs=hidden_layer, units=1 * self.embed_dim,
                                           kernel_regularizer=regularizers.l2_regularizer(0.001), activation=tf.nn.tanh,
                                           name='hidden_layer1')
            hidden_layer2 = tf.layers.dense(inputs=hidden_layer1, units=1 * self.embed_dim,
                                            kernel_regularizer=regularizers.l2_regularizer(0.001), activation=tf.nn.tanh,
                                            name='hidden_layer2')
            hidden_layer3 = tf.layers.dense(inputs=hidden_layer2, units=1 * self.embed_dim,
                                            kernel_regularizer=regularizers.l2_regularizer(0.001), activation=tf.nn.tanh,
                                            name='hidden_layer3')
            hidden_layer4 = tf.layers.dense(inputs=hidden_layer3, units=1 * self.embed_dim,
                                            kernel_regularizer=regularizers.l2_regularizer(0.001),
                                            activation=tf.nn.tanh,
                                            name='hidden_layer4')
            hidden_layer5 = tf.layers.dense(inputs=hidden_layer4, units=1 * self.embed_dim,
                                            kernel_regularizer=regularizers.l2_regularizer(0.001),
                                            activation=tf.nn.tanh,
                                            name='hidden_layer5')
            hidden_layer6 = tf.layers.dense(inputs=hidden_layer5, units=1 * self.embed_dim,
                                            kernel_regularizer=regularizers.l2_regularizer(0.001),
                                            activation=tf.nn.tanh,
                                            name='hidden_layer6')
            hidden_layer10 = tf.layers.dense(inputs=hidden_layer6, units=1 * self.embed_dim,
                                            kernel_regularizer=regularizers.l2_regularizer(0.001),
                                            activation=tf.nn.relu,
                                            name='hidden_layer10')
            attention_layer_all = self.embed_dim*tf.nn.softmax(hidden_layer10,axis=-1)

        attention_layer_pos, attention_layer = tf.split(attention_layer_all, [self.batch_size, self.batch_size*self.n_negative], 0)
        # positive item to user distance (N)
        pos_distances = tf.reduce_sum(tf.squared_difference(tf.multiply(attention_layer_pos,users),tf.multiply(attention_layer_pos,pos_items)), 1,name="pos_distances")
        attention_reshape = tf.transpose(tf.reshape(attention_layer, [-1, self.n_negative, self.embed_dim]), [0, 2, 1])
        # distance to negative items (N x W)
        distance_to_neg_items = tf.reduce_sum(
            tf.squared_difference(tf.multiply(attention_reshape, tf.expand_dims(users, -1)),
                                  tf.multiply(attention_reshape, tf.transpose(neg_items, [0, 2, 1]))), 1,
            name="distance_to_neg_items")
        logging.log(logging.INFO,'distance_to_neg_items.shape:'+str(distance_to_neg_items.shape))

        # best negative item (among W negative samples) their distance to the user embedding (N)
        closest_negative_item_distances = tf.reduce_min(distance_to_neg_items, 1, name="closest_negative_distances")
        logging.log(logging.INFO,'closest_negtive_item_distances:'+str(closest_negative_item_distances.shape))
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
        loss = self.embedding_loss
        if self.use_cov_loss:
            loss += self.covariance_loss
        total_loss = loss
        return total_loss

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
        user = tf.expand_dims(tf.nn.embedding_lookup(self.user_embeddings, self.score_user_ids), 1)
        items = tf.tile(tf.expand_dims(self.item_embeddings, 0),[tf.shape(user)[0],1,1])
        # (1, N_ITEM, K)
        input_concat = tf.concat([self.item_embeddings,self.t_features,self.v_features],1)
        #t_features = tf.tile(tf.expand_dims(self.t_features, 0),[tf.shape(user)[0],1,1])
        #v_features = tf.tile(tf.expand_dims(self.v_features, 0),[tf.shape(user)[0],1,1])

        with tf.variable_scope("concat_dense"):
            hidden_layer = tf.layers.dense(inputs=input_concat,units=8*self.embed_dim,activation=tf.nn.relu,name='hidden_layer',reuse=True)
            dropout = tf.layers.dropout(inputs=hidden_layer, rate=self.dropout_rate)
            hidden_layer1 = tf.layers.dense(inputs=dropout,units=4*self.embed_dim,activation=tf.nn.relu,name='hidden_layer1',reuse=True)
            dropout1 = tf.layers.dropout(inputs=hidden_layer1, rate=self.dropout_rate)
            hidden_layer2 = tf.layers.dense(inputs=dropout1,units=2*self.embed_dim,activation=tf.nn.relu,name='hidden_layer2',reuse=True)
            dropout2 = tf.layers.dropout(inputs=hidden_layer2, rate=self.dropout_rate)
            output = tf.layers.dense(inputs=dropout2,units=self.embed_dim,name='output',reuse=True)
        item = tf.tile(tf.expand_dims(output, 0),[tf.shape(user)[0],1,1])
        input = tf.concat(
                [tf.reshape(tf.tile(user, [1, tf.shape(item)[1], 1]), [-1, self.embed_dim]),
                 tf.reshape(item, [-1, self.embed_dim])], 1)
        with tf.variable_scope('dense'):
            hidden_layer = tf.layers.dense(inputs=input, units=1*self.embed_dim,trainable=False,
                                           kernel_regularizer=regularizers.l2_regularizer(0.001),activation=tf.nn.tanh,
                                           name='hidden_layer', reuse=True)
            hidden_layer1 = tf.layers.dense(inputs=hidden_layer, units=1 * self.embed_dim, trainable=False,
                                           kernel_regularizer=regularizers.l2_regularizer(0.001), activation=tf.nn.tanh,
                                           name='hidden_layer1', reuse=True)
            hidden_layer2 = tf.layers.dense(inputs=hidden_layer1, units=1 * self.embed_dim, trainable=False,
                                            kernel_regularizer=regularizers.l2_regularizer(0.001), activation=tf.nn.tanh,
                                            name='hidden_layer2', reuse=True)
            hidden_layer3 = tf.layers.dense(inputs=hidden_layer2, units=1 * self.embed_dim, trainable=False,
                                            kernel_regularizer=regularizers.l2_regularizer(0.001), activation=tf.nn.tanh,
                                            name='hidden_layer3', reuse=True)
            hidden_layer4 = tf.layers.dense(inputs=hidden_layer3, units=1 * self.embed_dim, trainable=False,
                                            kernel_regularizer=regularizers.l2_regularizer(0.001),
                                            activation=tf.nn.tanh,
                                            name='hidden_layer4', reuse=True)
            hidden_layer5 = tf.layers.dense(inputs=hidden_layer4, units=1 * self.embed_dim, trainable=False,
                                            kernel_regularizer=regularizers.l2_regularizer(0.001),
                                            activation=tf.nn.tanh,
                                            name='hidden_layer5', reuse=True)
            hidden_layer6 = tf.layers.dense(inputs=hidden_layer5, units=1 * self.embed_dim, trainable=False,
                                            kernel_regularizer=regularizers.l2_regularizer(0.001),
                                            activation=tf.nn.tanh,
                                            name='hidden_layer6', reuse=True)
            hidden_layer10 = tf.layers.dense(inputs=hidden_layer6, units=1 * self.embed_dim, trainable=False,
                                            kernel_regularizer=regularizers.l2_regularizer(0.001),
                                            activation=tf.nn.relu,
                                            name='hidden_layer10', reuse=True)
            attention_layer_score = self.embed_dim*tf.nn.softmax(hidden_layer10,axis=-1)

        attention_reshape = tf.reshape(attention_layer_score,[-1,tf.shape(item)[1],self.embed_dim])
        return -tf.reduce_sum(tf.squared_difference(tf.multiply(attention_reshape,user),tf.multiply(attention_reshape, items)), 2, name="scores")

BATCH_SIZE = 5000  #50000
N_NEGATIVE = 4
EVALUATION_EVERY_N_BATCHES = 400
EMBED_DIM = 64
hidden_layer_dim = 256
dropout_rate = 0.5,
eval_num = 30
Filename = 'Office'
Filepath = 'Data/'+ Filename

import pandas as pd
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

def optimize(model, sampler,train, test):
    """
    Optimize the model. TODO: implement early-stopping
    :param model: model to optimize
    :param sampler: mini-batch sampler
    :param train: train user-item matrix
    :param valid: validation user-item matrix
    :return: None
    """
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # sample some users to calculate recall validation
    test_users = numpy.asarray(list(set(test.nonzero()[0])), dtype=numpy.int32)
    epoch = 0
    tempbest = 0
    while True:
        print('\nepochs:'+str(epoch))
        epoch += 1
        # TODO: early stopping based on validation recall
        # train model
        losses = []
        # run n mini-batches
        for _ in range(EVALUATION_EVERY_N_BATCHES):
            user_pos, neg = sampler.next_batch()
            _, loss = sess.run((model.optimize, model.loss),
                               {model.user_positive_items_pairs: user_pos,
                                model.negative_samples: neg})
            losses.append(loss)
        print("\nTraining loss "+ str(numpy.mean(losses)))
        testresult = RecallEvaluator(model, train, test)
        # compute recall,ndcg,hr,pr on test set
        test_recalls = []
        test_ndcg = []
        test_hr = []
        test_pr = []
        for user_chunk in toolz.partition_all(50, test_users):
            recalls, ndcgs, hit_ratios, precisions = testresult.eval(sess, user_chunk)
            test_recalls.extend(recalls)
            test_ndcg.extend(ndcgs)
            test_hr.extend(hit_ratios)
            test_pr.extend(precisions)

        if sum(test_ndcg) / float(len(test_ndcg)) > tempbest:
            tempbest = sum(test_ndcg) / float(len(test_ndcg))
            print('________HR:tempbest__________:' + str(tempbest))
        print("\nresult on test set: ndcg:{},recall:{},hr:{},pr:{}".format(sum(test_ndcg) / float(len(test_ndcg)),
                                                                           sum(test_recalls) / float(len(test_recalls)),
                                                                           sum(test_hr) / float(len(test_hr)),
                                                                           sum(test_pr) / float(len(test_pr))))


if __name__ == '__main__':
    dftrain = pd.read_csv('Data/Office/train.csv', index_col=None, usecols=None)
    dftest = pd.read_csv('Data/Office/test.csv', index_col=None, usecols=None)
    trainlist = []
    testlist = []
    for index, row in dftrain.iterrows():
        u, i = int(row['userID']), int(row['itemID'])
        trainlist.append([u,i])
    for index, row in dftest.iterrows():
        u, i = int(row['userID']), int(row['itemID'])
        testlist.append([u,i])
    for i in range(len(testlist)):
        if testlist[i] in trainlist:
            print(testlist[i])
    #print(dftrain,dftest)
    # get user-item matrix
    #user_item_matrix, features = citeulike(tag_occurence_thres=5)
    # make feature as dense matrix
    #dense_features = features.toarray() + 1E-10
    # get train/valid/test user-item matrices
    dataset = Dataset.Dataset(Filepath)
    train, test= dataset.trainMatrix, dataset.testRatings

    t_features,v_features = dataset.textualfeatures,dataset.imagefeatures

    #user_similar,item_similar = dataset.user_similar,dataset.item_similar
    n_users, n_items = train.shape
    print(str(n_users)+','+str(n_items))
    # create warp sampler
    sampler = WarpSampler(train, batch_size=BATCH_SIZE, n_negative=N_NEGATIVE, check_negative=True)

    # WITHOUT features
    # Train a user-item joint embedding, where the items a user likes will be pulled closer to this users.
    # Once the embedding is trained, the recommendations are made by finding the k-Nearest-Neighbor to each user.
    model = CML(n_users,
                n_items,
                # set features to None to disable feature projection
                t_features=t_features,
                v_features=v_features,
                # size of embedding
                embed_dim=EMBED_DIM,
                dropout_rate=0.8,
                # batche_size
                batch_size=BATCH_SIZE,
                # N_negatvie
                n_negative = N_NEGATIVE,
                eval_num = eval_num,
                # the size of hinge loss margin.
                margin=1.6,
                # clip the embedding so that their norm <= clip_norm
                clip_norm=1.0,
                # learning rate for AdaGrad
                master_learning_rate=0.001,
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
                cov_loss_weight=1.0
                )
    optimize(model, sampler,train, test)
