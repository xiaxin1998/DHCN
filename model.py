import tensorflow as tf
import math
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class DHCN(object):
    def __init__(self, emb_size=100, batch_size=100, n_node=None,lr=None, l2=None, layers=None,beta=None):
        self.emb_size = emb_size
        self.batch_size = batch_size
        self.stdv = 1.0 / math.sqrt(self.emb_size)
        self.n_node = n_node
        self.L2 = l2
        self.layers = layers
        self.beta = beta
        self.tar = tf.placeholder(dtype=tf.int32)  # the true label of training set, e.g. the next item of each session
        self.H = tf.placeholder(dtype=tf.float32)  # the incidence matrix of hypergraph
        self.H_T = tf.placeholder(dtype=tf.float32)   # the tramsposed incidence matrix of hypergraph
        self.B = tf.placeholder(dtype=tf.float32)     # the hyperedge degree matrix of hypergrpah (B**-1)
        self.D = tf.placeholder(dtype=tf.float32)     # the node degree matrix of hypergrpah  (D**-1)
        self.item = tf.placeholder(dtype=tf.int32)    # the session sequence with new item id
        self.item_map = tf.placeholder(dtype=tf.int32)   # the item in this batch, shape:(n, 100), n is the item number of this batch
        self.session_len = tf.placeholder(dtype=tf.float32)   # the length of each session, shape:(100,)
        self.session_item = tf.placeholder(dtype=tf.int32)    # the raw session sequence
        self.last = tf.placeholder(dtype=tf.int32)            # the last item id of each session using the new item id
        self.A_hat = tf.placeholder(dtype=tf.float32)         # the incidence matrix of line graph (A**-1)
        self.D_hat = tf.placeholder(dtype=tf.float32)         # the edge degree matrix of line graph (D**-1)
        self.initializer = tf.random_uniform_initializer(-self.stdv, self.stdv)
        self.nasr_w1 = tf.get_variable('nasr_w1', [self.emb_size, self.emb_size], dtype=tf.float32, initializer=self.initializer)
        self.nasr_w2 = tf.get_variable('nasr_w2', [self.emb_size, self.emb_size], dtype=tf.float32, initializer=self.initializer)
        self.nasr_v = tf.get_variable('nasrv', [1, self.emb_size], dtype=tf.float32, initializer=self.initializer)
        self.nasr_b = tf.get_variable('nasr_b', [self.emb_size], dtype=tf.float32, initializer=tf.zeros_initializer())
        self.embedding = tf.get_variable(shape=[n_node, emb_size], name='embedding', dtype=tf.float32, initializer=self.initializer)

        self.weights = {}
        for k in range(self.layers):
            #transformation matrices
            self.weights['P_h%d' % (k+1)] = tf.get_variable(shape=[self.emb_size, self.emb_size], name='P_h%d' % (k+1), dtype=tf.float32, initializer=self.initializer)
            self.weights['P_l%d' % (k+1)] = tf.get_variable(shape=[self.emb_size, self.emb_size], name='P_l%d' % (k+1), dtype=tf.float32, initializer=self.initializer)
        for i in range(2):
            self.weights['gating%d' % (i)] = tf.get_variable('g_W_%d_1' % (i), [self.emb_size, self.emb_size], dtype=tf.float32, initializer=self.initializer)
            self.weights['gating_bias%d' % (i)] = tf.get_variable('g_W_b_%d_1' % (i), [1, self.emb_size],dtype=tf.float32, initializer=self.initializer)
        self.total_loss,  self.prediction = self.build()
        self.learning_rate = lr
        self.opt_rank = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        config = tf.ConfigProto(gpu_options=gpu_options)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

    def self_gating(self,emb,num):
        return tf.multiply(emb, tf.nn.sigmoid(tf.matmul(emb, self.weights['gating%d' % num]) + self.weights['gating_bias%d' % num]))

    def LineHypergraphConv(self):
        item_embeddings = self.self_gating(self.embedding,0)
        item_embeddings= tf.concat([tf.zeros([1, self.emb_size]), item_embeddings[1:]], 0)
        seq_embedding = tf.nn.embedding_lookup(item_embeddings, self.session_item)
        session_emb_lgcn = tf.divide(tf.reduce_sum(seq_embedding, axis=1), self.session_len)
        session = [session_emb_lgcn]
        DA = tf.matmul(self.D_hat, self.A_hat)
        for i in range(self.layers):
            session_emb_lgcn = tf.matmul(session_emb_lgcn, self.weights['P_l%d' % (i+1)])
            session_emb_lgcn = tf.matmul(DA, session_emb_lgcn)
            session_emb_lgcn = tf.nn.l2_normalize(session_emb_lgcn)
            session.append(session_emb_lgcn)
        session_emb_lgcn = tf.reduce_sum(session, 0)
        return session_emb_lgcn

    def HypergraphConv(self):
        item_embeddings = self.self_gating(self.embedding, 1)
        item_embeddings = tf.nn.embedding_lookup(item_embeddings, self.item_map)
        item_embedding_layer0 = item_embeddings
        DHW = tf.matmul(self.D, self.H)
        DHWBH = tf.matmul(DHW, tf.matmul(self.B, self.H_T))
        final = [item_embedding_layer0]
        for i in range(self.layers):
            item_embeddings = tf.matmul(item_embeddings, self.weights['P_h%d' % (i+1)])
            item_embeddings = tf.matmul(DHWBH, item_embeddings)
            item_embeddings = tf.nn.l2_normalize(item_embeddings)
            final.append(item_embeddings)
        item_embeddings = tf.reduce_sum(final, 0)
        return item_embeddings, item_embedding_layer0

    def generate_sess_emb(self,item_embedding):
        last_h = tf.nn.embedding_lookup(item_embedding, self.last)
        last = tf.matmul(last_h, self.nasr_w1)
        e = tf.concat([tf.zeros([1, self.emb_size]), item_embedding], 0)
        seq_h = tf.nn.embedding_lookup(e, self.item)
        seq = tf.matmul(tf.reshape(seq_h, [-1, self.emb_size]), self.nasr_w2)
        last = tf.reshape(last, [self.batch_size, 1, -1])
        m = tf.nn.sigmoid(last + tf.reshape(seq, [self.batch_size, -1, self.emb_size]) + self.nasr_b)
        coef = tf.matmul(tf.reshape(m, [-1, self.emb_size]), self.nasr_v, transpose_b=True)
        session_emb_hgnn = tf.concat([tf.reduce_sum(tf.reshape(coef, [self.batch_size, -1, 1]) * seq_h, 1),tf.reshape(last, [-1, self.emb_size])], -1)
        self.w = tf.get_variable('w', [2 * self.emb_size, self.emb_size],initializer=self.initializer)
        session_emb_hgnn = tf.matmul(session_emb_hgnn, self.w)
        return session_emb_hgnn

    def ranking_task(self, sess_emb_hgnn,sess_emb_lgcn):
        item = self.embedding[1:]
        logits = tf.matmul(sess_emb_hgnn, item, transpose_b=True) + tf.matmul(sess_emb_lgcn, item, transpose_b=True)
        loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.tar, logits=logits))
        self.vars = tf.trainable_variables()
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in self.vars]) * self.L2
        loss = loss + lossL2
        return loss, logits

    def SSL_task(self, sess_emb_hgnn, sess_emb_lgcn):
        def row_shuffle(embedding):
            corrupted_embedding = tf.gather(embedding, tf.random.shuffle(tf.range(tf.shape(embedding)[0])))
            return corrupted_embedding
        def row_column_shuffle(embedding):
            corrupted_embedding = tf.gather(embedding, tf.random.shuffle(tf.range(tf.shape(embedding)[0])))
            corrupted_embedding = tf.gather(tf.transpose(corrupted_embedding), tf.random.shuffle(tf.range(tf.shape(tf.transpose(corrupted_embedding))[0])))
            return tf.transpose(corrupted_embedding)
        def score(x1, x2):
            return tf.reduce_sum(tf.multiply(x1, x2), 1)
            
        sess_emb_hgnn = tf.nn.l2_normalize(sess_emb_hgnn, 1)
        sess_emb_lgcn = tf.nn.l2_normalize(sess_emb_lgcn, 1)
        # contrastive1 learning between sessions
        pos = score(sess_emb_hgnn, sess_emb_lgcn)
        neg1 = score(sess_emb_lgcn, row_column_shuffle(sess_emb_hgnn))
        neg2 = score(sess_emb_hgnn, row_column_shuffle(sess_emb_lgcn))
        con_loss = tf.reduce_sum(-tf.log(tf.sigmoid(pos))-tf.log(1-tf.sigmoid(neg1))-tf.log(1-tf.sigmoid(neg2)))
        return con_loss

    def build(self):
        self.item_embedding, self.item_embedding_layer0 = self.HypergraphConv()
        sess_emb_lgcn = self.LineHypergraphConv()
        sess_emb_hgnn = self.generate_sess_emb(self.item_embedding)
        self.rank_loss, self.logits = self.ranking_task(sess_emb_hgnn,sess_emb_lgcn)
        self.con_loss = self.SSL_task(sess_emb_hgnn, sess_emb_lgcn)
        self.loss = self.rank_loss + self.beta*self.con_loss
        return self.loss, self.logits

    def run(self, fetches, tar, items, last,H, H_T, D, B, item_map, session_len, session_item,A_hat, D_hat):
        return self.sess.run(fetches, feed_dict={self.tar: tar, self.item: items, self.last: last,
                                                 self.H: H, self.H_T: H_T, self.D: D, self.B: B,
                                                 self.A_hat: A_hat, self.D_hat: D_hat, self.item_map: item_map,
                                                 self.session_len: session_len, self.session_item: session_item})





