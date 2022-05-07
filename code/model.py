import tensorflow as tf
import numpy as np
import os

from util_se import _create_gcn_emb, _create_gat_emb, attn


class myModel(object):
    def __init__(self,
                 n_mid,
                 n_user,
                 embedding_dim,
                 batch_size,
                 seq_len,
                 neg_num
                 ):
        self.n_mid = n_mid
        self.n_user = n_user
        self.neg_num = neg_num
        self.batch_size = batch_size
        self.seq_len = seq_len

        with tf.name_scope('inputs'):
            self.mid_his_batch_ph = tf.placeholder(tf.int32, [None, None], name='mid_his_batch_ph')
            self.uid_batch_ph = tf.placeholder(tf.int32, [None, ], name='uid_batch_ph')
            self.mid_batch_ph = tf.placeholder(tf.int32, [None, ], name='mid_batch_ph')
            self.mask = tf.placeholder(tf.float32, [None, None], name='mask_batch_ph')
            self.label = tf.placeholder(tf.float32, [None, ], name='label_batch_ph')
            self.lr = tf.placeholder(tf.float64, [], name='lr')

        with tf.name_scope('Embedding_layer'):
            self.mid_embeddings_var = tf.get_variable('mid_embedding_var', [n_mid, embedding_dim], trainable=True)
            self.user_embeddings_var = tf.get_variable('uid_embedding_var', [n_user, embedding_dim], trainable=True,
                                                      initializer=tf.random_normal_initializer(0, 0.1))

            self.mid_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_batch_ph)
            self.mid_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_his_batch_ph)
            self.uid_batch_embedded = tf.nn.embedding_lookup(self.user_embeddings_var, self.uid_batch_ph)

        self.item_eb = self.mid_batch_embedded
        self.user_eb = self.uid_batch_embedded
        self.item_his_eb = self.mid_his_batch_embedded * tf.reshape(self.mask, (-1, seq_len, 1))

    def build_multi_ce_loss(self, item_emb, user_emb_list):
        res = []
        loss = []
        for i in range(self.num_layer):
            tmp = attn(item_emb, user_emb_list[i])
            res.append(tf.expand_dims(tmp, axis=1))
            logits_tmp = tf.reduce_sum(item_emb*tmp, axis=-1)
            loss.append(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.reshape(self.label, (-1, )), logits=tf.reshape(logits_tmp, (-1, )))))
        res_concated = tf.concat(res, axis=1)
        logits_ = tf.reduce_sum(tf.expand_dims(item_emb, axis=1) * res_concated, axis=-1)
        self.logits = tf.reduce_max(logits_, axis=-1)
        l2_loss = 1e-5 * tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()])
        self.loss = l2_loss + 1e-5*self.adj_l1#loss[0] + loss[1] + loss[2] + loss[3] + l2_loss + 1e-5*self.adj_l1
        for i in range(self.num_layer):
            self.loss += loss[i]
        #self.loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.reshape(self.label, (-1, )), logits=tf.reshape(self.logits, (-1, )))) + l2_loss
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def build_ce_loss(self, item_emb, user_emb):
        res = attn(item_emb, user_emb)
        weight_decay = 1e-5
        l2_loss = weight_decay * tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()]) 
        self.logits = tf.reduce_sum(item_emb * res, axis=-1)
        self.loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.reshape(self.label, (-1, )), logits=tf.reshape(self.logits, (-1, )))) + l2_loss
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def train(self, sess, inps):
        feed_dict = {
            self.uid_batch_ph: inps[0],
            self.mid_batch_ph: inps[1],
            self.mid_his_batch_ph: inps[2],
            self.mask: inps[3],
            self.label: inps[4],
            self.lr: inps[5],
        }
        loss, _ = sess.run([self.loss, self.optimizer], feed_dict=feed_dict)
        return loss

    def test(self, sess, inps):
        feed_dict = {
            self.uid_batch_ph: inps[0],
            self.mid_batch_ph: inps[1],
            self.mid_his_batch_ph: inps[2],
            self.mask: inps[3],
        }
        logits = sess.run(self.logits, feed_dict=feed_dict)
        return logits

    def save(self, sess, path):
        if not os.path.exists(path):
            os.makedirs(path)
        saver = tf.train.Saver()
        saver.save(sess, path + 'model.ckpt')

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, path + 'model.ckpt')
        print('model restored from %s' % path)


def get_shape(inputs):
    dynamic_shape = tf.shape(inputs)
    static_shape = inputs.get_shape().as_list()
    shape = []
    for i, dim in enumerate(static_shape):
        shape.append(dim if dim is not None else dynamic_shape[i])

    return shape


class CapsuleNetwork(tf.layers.Layer):
    def __init__(self, dim, seq_len, bilinear_type=2, num_interest=4, hard_readout=True, relu_layer=False):
        super(CapsuleNetwork, self).__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.bilinear_type = bilinear_type
        self.num_interest = num_interest
        self.hard_readout = hard_readout
        self.relu_layer = relu_layer
        self.stop_grad = True
        print('bilinear_type:', bilinear_type)

    def _birnn(self, x):
        x = tf.unstack(x, self.seq_len, 1)
        with tf.variable_scope("bilstm", reuse=tf.AUTO_REUSE):
            lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(self.dim)
            lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(self.dim)
            outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)

            output = tf.concat(outputs, axis=0)
            output = tf.reshape(output, (-1, self.seq_len, self.dim*2))
            output = tf.layers.dense(output, self.dim*self.num_interest, activation=None, use_bias=False)
        return output

    def _rnn(self, x):
        with tf.variable_scope("lstm", reuse=tf.AUTO_REUSE):
            lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(self.dim*2, forget_bias=1.0)
        outputs, _, = tf.nn.dynamic_rnn(lstm_fw_cell, x, dtype=tf.float32)

        output = tf.concat(outputs, axis=0)
        output = tf.reshape(output, (-1, self.seq_len, self.dim*2))
        output = tf.layers.dense(output, self.dim*self.num_interest, activation=None, use_bias=False)
        return output


    def call(self, item_his_emb, item_eb, mask):
        with tf.variable_scope('bilinear'):
            if self.bilinear_type == 0:
                item_emb_hat = tf.layers.dense(item_his_emb, self.dim, activation=None, bias_initializer=None)
                item_emb_hat = tf.tile(item_emb_hat, [1, 1, self.num_interest])
            elif self.bilinear_type == 1:
                #item_emb_hat = tf.layers.dense(item_his_emb, self.dim * self.num_interest, activation=None, bias_initializer=None)
                item_emb_hat = self._rnn(tf.reverse(item_his_emb, [1]))        
            else:
                w = tf.get_variable(
                    'weights', shape=[1, self.seq_len, self.num_interest * self.dim, self.dim],
                    initializer=tf.random_normal_initializer())
                # [N, T, 1, C]
                u = tf.expand_dims(item_his_emb, axis=2)
                # [N, T, num_caps * dim_caps]
                item_emb_hat = tf.reduce_sum(w[:, :self.seq_len, :, :] * u, axis=3)
                #item_emb_hat_t = self._birnn(item_his_emb)
        item_emb_hat = tf.reshape(item_emb_hat, [-1, self.seq_len, self.num_interest, self.dim])
        item_emb_hat = tf.transpose(item_emb_hat, [0, 2, 1, 3])
        #item_emb_hat_t = tf.reshape(item_emb_hat_t, [-1, self.num_interest, self.seq_len, self.dim])

        if self.stop_grad:
            item_emb_hat_iter = tf.stop_gradient(item_emb_hat, name='item_emb_hat_iter')
        else:
            item_emb_hat_iter = item_emb_hat

        capsule_weight = tf.stop_gradient(tf.truncated_normal([get_shape(item_his_emb)[0], self.num_interest, self.seq_len], stddev=1.0))

        for i in range(3):
            atten_mask = tf.tile(tf.expand_dims(mask, axis=1), [1, self.num_interest, 1])
            paddings = tf.zeros_like(atten_mask)

            capsule_softmax_weight = tf.nn.softmax(capsule_weight, axis=1)
            capsule_softmax_weight = tf.where(tf.equal(atten_mask, 0), paddings, capsule_softmax_weight)
            capsule_softmax_weight = tf.expand_dims(capsule_softmax_weight, 2)

            if i < 2:
                interest_capsule = tf.matmul(capsule_softmax_weight, item_emb_hat_iter)
                cap_norm = tf.reduce_sum(tf.square(interest_capsule), -1, True)
                scalar_factor = cap_norm / (1 + cap_norm) / tf.sqrt(cap_norm + 1e-9)
                interest_capsule = scalar_factor * interest_capsule

                delta_weight = tf.matmul(item_emb_hat_iter, tf.transpose(interest_capsule, [0, 1, 3, 2]))
                delta_weight = tf.reshape(delta_weight, [-1, self.num_interest, self.seq_len])
		
                #delta_weight_t = tf.matmul(item_emb_hat_iter_t, tf.transpose(interest_capsule, [0, 1, 3, 2]))
                #delta_weight_t = tf.reshape(delta_weight_t, [-1, self.num_interest, self.seq_len])
                capsule_weight = capsule_weight + delta_weight
                if i == 0:
                    item_emb_hat_iter = self._birnn(tf.reshape(tf.stop_gradient(item_emb_hat_iter), [-1, self.seq_len, self.num_interest*self.dim]))
                    item_emb_hat_iter = tf.reshape(item_emb_hat_iter, [-1, self.num_interest, self.seq_len, self.dim]) + tf.stop_gradient(item_emb_hat)
                else:
                    item_emb_hat_iter = tf.stop_gradient(item_emb_hat)
            else:
                interest_capsule = tf.matmul(capsule_softmax_weight, item_emb_hat)
                cap_norm = tf.reduce_sum(tf.square(interest_capsule), -1, True)
                scalar_factor = cap_norm / (1 + cap_norm) / tf.sqrt(cap_norm + 1e-9)
                interest_capsule = scalar_factor * interest_capsule

        interest_capsule = tf.reshape(interest_capsule, [-1, self.num_interest, self.dim])

        if self.relu_layer:
            interest_capsule = tf.layers.dense(interest_capsule, self.dim, activation=tf.nn.relu, name='proj')

        atten = tf.matmul(interest_capsule, tf.reshape(item_eb, [-1, self.dim, 1]))
        atten = tf.nn.softmax(tf.pow(tf.reshape(atten, [-1, self.num_interest]), 1))

        if self.hard_readout:
            readout = tf.gather(tf.reshape(interest_capsule, [-1, self.dim]), tf.argmax(atten, axis=1, output_type=tf.int32) + tf.range(tf.shape(item_his_emb)[0]) * self.num_interest)
        else:
            readout = tf.matmul(tf.reshape(atten, [get_shape(item_his_emb)[0], 1, self.num_interest]), interest_capsule)
            readout = tf.reshape(readout, [get_shape(item_his_emb)[0], self.dim])

        return interest_capsule, readout


def normalize_adj_tensor(adj, seq_len):
    adj = adj + tf.expand_dims(tf.eye(seq_len), axis=0)
    rowsum = tf.reduce_sum(adj, axis=1)
    d_inv_sqrt = tf.pow(rowsum, -0.5)
    candidate_a = tf.zeros_like(d_inv_sqrt)
    d_inv_sqrt = tf.where(tf.math.is_inf(d_inv_sqrt), candidate_a, d_inv_sqrt)
    d_mat_inv_sqrt = tf.matrix_diag(d_inv_sqrt)
    norm_adg = tf.matmul(d_mat_inv_sqrt, adj)
    return norm_adg


class modelTy(myModel):
    def __init__(self,
                 n_mid,
                 n_user,
                 embedding_dim,
                 batch_size,
                 seq_len,
                 neg_num,
                 hidden_size,
                 num_interest,
                 num_layer,
                 se_num,
                 norm_adj=False,
                 hard_readout=True,
                 relu_layer=False
                 ):
        super(modelTy, self).__init__(n_mid,
                                      n_user,
                                      embedding_dim,
                                      batch_size,
                                      seq_len+se_num,
                                      neg_num)
        self.num_layer = num_layer
        adj_l = tf.tile(tf.expand_dims(self.item_his_eb, axis=2), [1, 1, seq_len+se_num, 1])
        adj_r = tf.tile(tf.expand_dims(self.item_his_eb, axis=1), [1, seq_len+se_num, 1, 1])
        
        # whether apply user_emb
        if 1:
            adj = tf.nn.sigmoid(tf.reduce_sum(adj_l * adj_r, axis=-1))#tf.concat([adj_l, adj_r], axis=3)
            adj = adj * tf.expand_dims(self.mask, axis=1)
            adj = adj * tf.expand_dims(self.mask, axis=2)

        else:
	        adj_node = tf.multiply(adj_l, adj_r)
	        adj_user = tf.expand_dims(tf.expand_dims(self.user_eb, axis=1), axis=2)
	        adj = tf.nn.sigmoid(tf.reduce_sum(adj_node*adj_user, axis=-1))
	        adj = adj * tf.expand_dims(self.mask, axis=1)
	        adj = adj * tf.expand_dims(self.mask, axis=2)

        self.adj_l1 = tf.norm(adj, ord=1)

        if norm_adj:
            adj = normalize_adj_tensor(adj, seq_len) 
        #all_embedding = [_create_gat_emb(self.mid_his_batch_embedded, [4, 1], [seq_len, embedding_dim], 0.2, 0.2, adj)]
        all_embedding = _create_gcn_emb(adj, self.mid_his_batch_embedded, num_layer-1, embedding_dim, se_num, batch_size, seq_len, layer_size=[embedding_dim, embedding_dim, embedding_dim])
        #self.item_his_eb = all_embedding[-1] * tf.reshape(self.mask, (-1, seq_len, 1))

        capsule_network = CapsuleNetwork(hidden_size, seq_len, bilinear_type=2, num_interest=num_interest, hard_readout=hard_readout, relu_layer=relu_layer)
        user_eb_list = []
        mask_new = tf.slice(self.mask, [0, se_num], [batch_size, seq_len])
        for l in range(num_layer):
            user_eb, _ = capsule_network(all_embedding[l], self.item_eb, mask_new)
            user_eb_list.append(user_eb)
        
        
        #self.build_ce_loss(self.item_eb, tf.concat(user_eb_list, axis=1))
        

        if len(user_eb_list) == 1:
            self.build_ce_loss(self.item_eb, user_eb_list[0])
        else:
            self.build_multi_ce_loss(self.item_eb, user_eb_list)
