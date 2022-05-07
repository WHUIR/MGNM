import tensorflow as tf
import time
import math
from sklearn.metrics import roc_auc_score


import numpy as np

def ln(inputs, epsilon=1e-8, scope="ln"):
    '''Applies layer normalization. See https://arxiv.org/abs/1607.06450.
    inputs: A tensor with 2 or more dimensions, where the first dimension has `batch_size`.
    epsilon: A floating number. A very small number for preventing ZeroDivision Error.
    scope: Optional scope for `variable_scope`.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
        gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs

def scaled_dot_product_attention(Q, K, V, key_masks,
                                 causality=False, dropout_rate=0.,
                                 training=True,
                                 scope="scaled_dot_product_attention"):
    '''See 3.2.1.
    Q: Packed queries. 3d tensor. [N, T_q, d_k].
    K: Packed keys. 3d tensor. [N, T_k, d_k].
    V: Packed values. 3d tensor. [N, T_k, d_v].
    key_masks: A 2d tensor with shape of [N, key_seqlen]
    causality: If True, applies masking for future blinding
    dropout_rate: A floating point number of [0, 1].
    training: boolean for controlling droput
    scope: Optional scope for `variable_scope`.
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        d_k = Q.get_shape().as_list()[-1]

        # dot product
        outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # (N, T_q, T_k)

        # scale
        outputs /= d_k ** 0.5

        # # key masking
        # outputs = mask(outputs, key_masks=key_masks, type="key")
        #
        # # causality or future blinding masking
        # if causality:
        #     outputs = mask(outputs, type="future")

        # softmax
        outputs = tf.nn.softmax(outputs)
        attention = tf.transpose(outputs, [0, 2, 1])
        tf.summary.image("attention", tf.expand_dims(attention[:1], -1))

        # # query masking
        # outputs = mask(outputs, Q, K, type="query")

        # dropout
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)

        # weighted sum (context vectors)
        outputs = tf.matmul(outputs, V)  # (N, T_q, d_v)

    return outputs

def multihead_attention(queries, keys, values, key_masks,
                        num_heads=8,
                        dropout_rate=0,
                        training=True,
                        causality=False,
                        scope="multihead_attention"):
    '''Applies multihead attention. See 3.2.2
    queries: A 3d tensor with shape of [N, T_q, d_model].
    keys: A 3d tensor with shape of [N, T_k, d_model].
    values: A 3d tensor with shape of [N, T_k, d_model].
    key_masks: A 2d tensor with shape of [N, key_seqlen]
    num_heads: An int. Number of heads.
    dropout_rate: A floating point number.
    training: Boolean. Controller of mechanism for dropout.
    causality: Boolean. If true, units that reference the future are masked.
    scope: Optional scope for `variable_scope`.

    Returns
      A 3d tensor with shape of (N, T_q, C)
    '''
    d_model = queries.get_shape().as_list()[-1]
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # Linear projections
        Q = tf.layers.dense(queries, d_model, use_bias=True)  # (N, T_q, d_model)
        K = tf.layers.dense(keys, d_model, use_bias=True)  # (N, T_k, d_model)
        V = tf.layers.dense(values, d_model, use_bias=True)  # (N, T_k, d_model)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, d_model/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, d_model/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, d_model/h)

        # Attention
        outputs = scaled_dot_product_attention(Q_, K_, V_, key_masks, causality, dropout_rate, training)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, d_model)

        # Residual connection
        outputs += queries

        # Normalize
        outputs = ln(outputs)

    return outputs


def ff(inputs, num_units, scope="positionwise_feedforward"):
    '''position-wise feed forward net. See 3.3

    inputs: A 3d tensor with shape of [N, T, C].
    num_units: A list of two integers.
    scope: Optional scope for `variable_scope`.
    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # Inner layer
        outputs = tf.layers.dense(inputs, num_units[0], activation=tf.nn.relu)

        # Outer layer
        outputs = tf.layers.dense(outputs, num_units[1])

        # Residual connection
        outputs += inputs

        # Normalize
        outputs = ln(outputs)

    return outputs


def label_smoothing(inputs, epsilon=0.1):
    '''Applies label smoothing. See 5.4 and https://arxiv.org/abs/1512.00567.
    inputs: 3d tensor. [N, T, V], where V is the number of vocabulary.
    epsilon: Smoothing rate.

    For example,

    ```
    import tensorflow as tf
    inputs = tf.convert_to_tensor([[[0, 0, 1],
       [0, 1, 0],
       [1, 0, 0]],
      [[1, 0, 0],
       [1, 0, 0],
       [0, 1, 0]]], tf.float32)

    outputs = label_smoothing(inputs)

    with tf.Session() as sess:
        print(sess.run([outputs]))

    >>
    [array([[[ 0.03333334,  0.03333334,  0.93333334],
        [ 0.03333334,  0.93333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334]],
       [[ 0.93333334,  0.03333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334],
        [ 0.03333334,  0.93333334,  0.03333334]]], dtype=float32)]
    ```
    '''
    V = inputs.get_shape().as_list()[-1]  # number of channels
    return ((1 - epsilon) * inputs) + (epsilon / V)


def positional_encoding(inputs,
                        maxlen,
                        masking=False,
                        scope="positional_encoding"):
    '''Sinusoidal Positional_Encoding. See 3.5
    inputs: 3d tensor. (N, T, E)
    maxlen: scalar. Must be >= T
    masking: Boolean. If True, padding positions are set to zeros.
    scope: Optional scope for `variable_scope`.
    returns
    3d tensor that has the same shape as inputs.
    '''

    E = inputs.get_shape().as_list()[-1]  # static
    N, T = tf.shape(inputs)[0], tf.shape(inputs)[1]  # dynamic
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # position indices
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])  # (N, T)

        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, (i - i % 2) / E) for i in range(E)]
            for pos in range(maxlen)])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
        position_enc = tf.convert_to_tensor(position_enc, tf.float32)  # (maxlen, E)

        # lookup
        outputs = tf.nn.embedding_lookup(position_enc, position_ind)

        # masks
        if masking:
            outputs = tf.where(tf.equal(inputs, 0), inputs, outputs)

        return tf.to_float(outputs)


def noam_scheme(init_lr, global_step, warmup_steps=4000.):
    '''Noam scheme learning rate decay
    init_lr: initial learning rate. scalar.
    global_step: scalar.
    warmup_steps: scalar. During warmup_steps, learning rate increases
        until it reaches init_lr.
    '''
    step = tf.cast(global_step + 1, dtype=tf.float32)
    return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)


def calculate_group_metric(labels, preds, users, calc_gauc=True, calc_ndcg=True, calc_hit=True, calc_mrr=True,
                           at_Ns=None):
    if at_Ns is None:
        at_Ns = [5]
    metrics = {}

    user_pred_dict = {}

    print_time_cost = False

    for i in range(len(users)):
        if users[i] in user_pred_dict:
            user_pred_dict[users[i]][0].append(preds[i])
            user_pred_dict[users[i]][1].append(labels[i])
        else:
            user_pred_dict[users[i]] = [[preds[i]], [labels[i]]]

    if calc_gauc:
        t = time.time()
        user_aucs = []
        valid_sample_num = 0
        for u in user_pred_dict:
            if 1 in user_pred_dict[u][1] and 0 in user_pred_dict[u][1]:  # contains both labels
                user_aucs.append(len(user_pred_dict[u][1]) * roc_auc_score(user_pred_dict[u][1], user_pred_dict[u][0]))
                valid_sample_num = len(user_pred_dict[u][1]) + valid_sample_num
        valid_group_num = len(user_aucs)+1
        total_group_num = len(user_pred_dict)+1
        total_sample_num = len(labels)+1
        metrics['gauc'] = (
            sum(user_aucs) / valid_sample_num)
        if print_time_cost:
            print("GAUC TIME: %.4fs" % (time.time() - t))

    t = time.time()
    if calc_ndcg or calc_hit or calc_mrr:
        for user, val in user_pred_dict.items():
            idx = np.argsort(val[0])[::-1]
            user_pred_dict[user][0] = np.array(val[0])[idx]
            user_pred_dict[user][1] = np.array(val[1])[idx]

    if calc_ndcg or calc_hit or calc_mrr:
        ndcg = np.zeros(len(at_Ns))
        hit = np.zeros(len(at_Ns))
        mrr = np.zeros(len(at_Ns))
        valid_user = 0
        for u in user_pred_dict:
            if 1 in user_pred_dict[u][1] and 0 in user_pred_dict[u][1]:  # contains both labels
                valid_user += 1
                pred = user_pred_dict[u][1]
                rank = np.nonzero(pred)[0][0]
                # print(pred, rank)
                for idx, n in enumerate(at_Ns):
                    if rank < n:
                        ndcg[idx] += 1 / np.log2(rank + 2)
                        hit[idx] += 1
                        mrr[idx] += 1 / (rank + 1)
        ndcg = ndcg / valid_user
        hit = hit / valid_user
        mrr = mrr / valid_user
        metrics['ndcg'] = ndcg
        metrics['hit'] = hit
        metrics['mrr'] = mrr
        if print_time_cost:
            print("NDCG TIME: %.4fs" % (time.time() - t))
    return metrics


def _create_gcn_emb(A, x, num_layer, embedding_dim, se_num, batch_size, seq_len, layer_size=[64, 64, 64]):
    initializer = tf.random_normal_initializer(stddev=0.01)
    weights_size_list = [embedding_dim] + layer_size
    all_weights = {}
    with tf.variable_scope("weights", reuse=tf.AUTO_REUSE):
        for lay in range(1):
            all_weights['W_gc%d' % lay] = tf.Variable(
                initializer([weights_size_list[lay], weights_size_list[lay+1]]), name='W_gc%d'%lay
            )
            all_weights['B_gc%d' % lay] = tf.Variable(
                initializer([1, weights_size_list[lay+1]]), name='b_gc%d'%lay
            )

    # gcn has three layers
    all_embeddings = [tf.slice(x, [0, se_num, 0], [batch_size, seq_len, embedding_dim])]
    for k in range(num_layer):
        embeddings = tf.matmul(A, x)
        embeddings = tf.nn.leaky_relu(tf.matmul(embeddings, all_weights['W_gc%d' % 0]) + all_weights['B_gc%d' % 0])
        all_embeddings.append(tf.slice(embeddings, [0, se_num, 0], [batch_size, seq_len, embedding_dim]))

    return all_embeddings


def _create_gat_emb(inputs, n_heads, hid_units, attn_drop, ffd_drop, bias_mat, activation=tf.nn.elu, residual=False):
    attns = []
    for _ in range(n_heads[0]):
        attns.append(attn_head(inputs, bias_mat=bias_mat,
                               out_sz=hid_units[0], activation=activation,
                               in_drop=ffd_drop,
                               coef_drop=attn_drop, residual=False))
    h1 = tf.concat(attns, axis=-1)
    for i in range(1, len(hid_units)):
        attns = []
        for _ in range(n_heads[i]):
            attns.append(attn_head(h1, bias_mat=bias_mat,
                                   out_sz=hid_units[i], activation=activation,
                                   in_drop=ffd_drop, coef_drop=attn_drop, residual=residual))
        h1 = tf.concat(attns, axis=-1)
    outs = []
    for i in range(n_heads[-1]):
        outs.append(attn_head(h1, bias_mat=bias_mat,
                                   out_sz=hid_units[-1], activation=activation,
                                   in_drop=ffd_drop, coef_drop=attn_drop, residual=residual))
        logits = tf.add_n(outs) / n_heads[-1]
    return logits

import numpy as np
import tensorflow as tf

conv1d = tf.layers.conv1d


def attn_head(seq, out_sz, bias_mat, activation, in_drop=0.0, coef_drop=0.0, residual=False):
    with tf.name_scope('my_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)

        # simplest self-attention possible
        f_1 = tf.layers.conv1d(seq_fts, 1, 1)
        f_2 = tf.layers.conv1d(seq_fts, 1, 1)
        logits = f_1 + tf.transpose(f_2, [0, 2, 1])
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)

        if coef_drop != 0.0:
            coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        vals = tf.matmul(coefs, seq_fts)
        ret = tf.contrib.layers.bias_add(vals)

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(seq, ret.shape[-1], 1)  # activation
            else:
                ret = ret + seq

        return activation(ret)  # activation


# Experimental sparse attention head (for running on datasets such as Pubmed)
# N.B. Because of limitations of current TF implementation, will work _only_ if batch_size = 1!
def sp_attn_head(seq, out_sz, adj_mat, activation, nb_nodes, in_drop=0.0, coef_drop=0.0, residual=False):
    with tf.name_scope('sp_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)

        # simplest self-attention possible
        f_1 = tf.layers.conv1d(seq_fts, 1, 1)
        f_2 = tf.layers.conv1d(seq_fts, 1, 1)

        f_1 = tf.reshape(f_1, (nb_nodes, 1))
        f_2 = tf.reshape(f_2, (nb_nodes, 1))

        f_1 = adj_mat * f_1
        f_2 = adj_mat * tf.transpose(f_2, [1, 0])

        logits = tf.sparse_add(f_1, f_2)
        lrelu = tf.SparseTensor(indices=logits.indices,
                                values=tf.nn.leaky_relu(logits.values),
                                dense_shape=logits.dense_shape)
        coefs = tf.sparse_softmax(lrelu)

        if coef_drop != 0.0:
            coefs = tf.SparseTensor(indices=coefs.indices,
                                    values=tf.nn.dropout(coefs.values, 1.0 - coef_drop),
                                    dense_shape=coefs.dense_shape)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        # As tf.sparse_tensor_dense_matmul expects its arguments to have rank-2,
        # here we make an assumption that our input is of batch size 1, and reshape appropriately.
        # The method will fail in all other cases!
        coefs = tf.sparse_reshape(coefs, [nb_nodes, nb_nodes])
        seq_fts = tf.squeeze(seq_fts)
        vals = tf.sparse_tensor_dense_matmul(coefs, seq_fts)
        vals = tf.expand_dims(vals, axis=0)
        vals.set_shape([1, nb_nodes, out_sz])
        ret = tf.contrib.layers.bias_add(vals)

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(seq, ret.shape[-1], 1)  # activation
            else:
                ret = ret + seq

        return activation(ret)  # activation


"""
 Prepare adjacency matrix by expanding up to a given neighbourhood.
 This will insert loops on every node.
 Finally, the matrix is converted to bias vectors.
 Expected shape: [graph, nodes, nodes]
"""

def adj_to_bias_tensor(adj, batch_size, row_size=20, nhood=1):
    nb_graphs = batch_size
    mt = tf.zeros_like(adj)
    for g in range(nb_graphs):
        mt[g] = tf.eye(row_size)
        for _ in range(nhood):
            mt[g] = tf.matmul(mt[g], (adj[g] + tf.eye(row_size)))
        for i in range(row_size):
            for j in range(row_size):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    return -1e9 * (1.0 - mt)


def attn(query, key):
    # key [batch_size, short_seq_length, units]
    # query [batch_size, units]
    alpha = tf.expand_dims(tf.nn.softmax(tf.reduce_sum(key * tf.expand_dims(query, axis=1), axis=-1), axis=-1), axis=-1) # [b_s, shrot_seq_length, 1]
    res = tf.reduce_sum(key * alpha, axis=1)
    return res

class InvLinear():
    r"""Permutation invariant linear layer.
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: ``True``
        reduction: Permutation invariant operation that maps the input set into a single
            vector. Currently, the following are supported: mean, sum, max and min.
    """
    def __init__(self, in_features, out_features, bias=False, reduction='mean'):
        super(InvLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        assert reduction in ['mean', 'sum', 'max', 'min'],  \
            '\'reduction\' should be \'mean\'/\'sum\'\'max\'/\'min\', got {}'.format(reduction)
        self.reduction = reduction

        self.beta = tf.get_variable(name='beta', shape=(self.in_features, self.out_features),
                                    initializer=tf.glorot_uniform_initializer())
        # self.beta = nn.Parameter(torch.Tensor(self.in_features,
        #                                       self.out_features))
        if bias:
            self.bias = tf.get_variable(name='bias', shape=(1, self.out_features), initializer=tf.random_uniform_initializer())
            # self.bias = nn.Parameter(torch.Tensor(1, self.out_features))

        # self.reset_parameters()

    # def reset_parameters(self):
    #     init.xavier_uniform_(self.beta)
    #     if self.bias is not None:
    #         fan_in, _ = init._calculate_fan_in_and_fan_out(self.beta)
    #         bound = 1 / math.sqrt(fan_in)
    #         init.uniform_(self.bias, -bound, bound)

    def forward(self, X, N, M, mask=None):
        r"""
        Maps the input set X = {x_1, ..., x_M} to a vector y of dimension out_features,
        through a permutation invariant linear transformation of the form:
            $y = \beta reduction(X) + bias$
        Inputs:
        X: N sets of size at most M where each element has dimension in_features
           (tensor with shape (N, M, in_features))
        mask: binary mask to indicate which elements in X are valid (byte tensor
            with shape (N, M) or None); if None, all sets have the maximum size M.
            Default: ``None``.
        Outputs:
        Y: N vectors of dimension out_features (tensor with shape (N, out_features))
        """

        y = tf.zeros((N, self.out_features))
        if mask is None:
            mask = tf.ones((N, M))

        if self.reduction == 'mean':
            sizes = tf.expand_dims(tf.reduce_sum(mask, axis=1), axis=1)
            Z = X * tf.expand_dims(mask, axis=2)
            y = tf.matmul(tf.reduce_sum(Z, axis=1), self.beta) / sizes

        elif self.reduction == 'sum':
            Z = X * tf.expand_dims(mask, axis=2)
            y = tf.matmul(tf.reduce_sum(Z, axis=1), self.beta)
        
        #elif self.reduction == 'max':
        #    Z = X
        #    Z[~mask] = float('-Inf')
        #    y = Z.max(dim=1)[0] @ self.beta
        #
        # else:  # min
        #     Z = X.clone()
        #     Z[~mask] = float('Inf')
        #     y = Z.min(dim=1)[0] @ self.beta

        #if self.bias is not None:
        #    y += self.bias

        return y

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, reduction={}'.format(
            self.in_features, self.out_features,
            self.bias is not None, self.reduction)

class BM25(object):
    def __init__(self, docs):
        self.D = len(docs)
        self.avgdl = sum([len(doc)+0.0 for doc in docs]) / self.D
        self.docs = docs
        self.f = []  # 列表的每一个元素是一个dict，dict存储着一个文档中每个词的出现次数
        self.df = {} # 存储每个词及出现了该词的文档数量
        self.idf = {} # 存储每个词的idf值
        self.k1 = 1.5
        self.b = 0.75
        self.init()

    def init(self):
        for doc in self.docs:
            tmp = {}
            for word in doc:
                tmp[word] = tmp.get(word, 0) + 1  # 存储每个文档中每个词的出现次数
            self.f.append(tmp)
            for k in tmp.keys():
                self.df[k] = self.df.get(k, 0) + 1
        for k, v in self.df.items():
            self.idf[k] = math.log(self.D-v+0.5)-math.log(v+0.5)

    def sim(self, doc, index):
        score = 0
        for word in doc:
            if word not in self.f[index]:
                continue
            d = len(self.docs[index])
            score += (self.idf[word]*self.f[index][word]*(self.k1+1)
                      / (self.f[index][word]+self.k1*(1-self.b+self.b*d
                                                      / self.avgdl)))
        return score
    def simall(self, doc):
        scores = []
        for index in range(self.D):
            score = self.sim(doc, index)
            scores.append(score)
        return scores

class Sortnovel(object):
    def __init__(self, docs):
        self.docs = docs  #所有文档列表,词表示
        self.bm25 = BM25(docs)

    def top(self, query):
        # [i1, i2, i3, ...]
        self.top = list(enumerate(self.bm25.simall(query)))
        self.sorttop = sorted(self.top, key=lambda x: x[1], reverse=True) #排序,匿名函数
        i = 0
        self.list = list(map(lambda x: x[0], self.sorttop))
        # print(self.list)    #输出序号
        # for index in self.list:     #输出id，书名，得分
        #     print(self.docs[index].novelid, self.novels[index].novelname, self.sorttop[i][1])
        #     i += 1
        return self.list

    def top1(self, query, limit = 10):
        self.top = list(enumerate(self.bm25.simall(query)))
        self.sorttop = sorted(self.top, key=lambda x: x[1], reverse=True) #排序,匿名函数
        i = 0
        self.list = list(map(lambda x: x[0], self.sorttop))[:limit]
        # print(self.list)
        # for index in self.list[:limit]:
        #     print(self.novels[index].novelid, self.novels[index].novelname, self.sorttop[i][1])
        #     i += 1
        return self.list


class Transformer:
    def __init__(self, num_heads, dimension, seq_len, dropout_rate=0.2, num_blocks=1):
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.num_blocks = num_blocks
        self.dimension = dimension
        self.seq_len = seq_len

    def encode(self, xs, mask, training=True):
        '''
        xs: [B, seq_len, D]
        Returns
        memory: encoder outputs. (N, T1, d)
        '''
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            xs += positional_encoding(xs, self.seq_len)
            xs = tf.layers.dropout(xs, self.dropout_rate, training=training)

            ## Blocks
            for i in range(self.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # self-attention
                    xs = multihead_attention(queries=xs,
                                              keys=xs,
                                              values=xs,
                                              key_masks=mask,
                                              num_heads=self.num_heads,
                                              dropout_rate=self.dropout_rate,
                                              training=training,
                                              causality=False)
                    # feed forward
                    xs = ff(xs, num_units=[self.dimension*4, self.dimension*4])
        memory = xs
        return memory, mask
