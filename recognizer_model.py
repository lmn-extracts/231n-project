'''
Holds code for the Recognizer Model
'''

from modules import *
from data_utils import *
import time
from data_provider import TFRecordReader
from dataset_helper import *

class Recognizer():
    def __init__(self, FLAGS):
        self.batch_size = FLAGS.batch_size
        self.height = FLAGS.height
        self.width = FLAGS.width
        self.hidden_size = FLAGS.hidden_size
        self.alphabet_size = FLAGS.alphabet_size
        self.FLAGS = FLAGS

        mprint('here','here')

        with tf.variable_scope('Recognizer_Model'):
            self.add_placeholders()
            self.build_graph()
            self.add_ctc_loss()

        # Define trainable parameters, gradient, gradient norm, and clip by gradient norm
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        self.gradient_norm = tf.global_norm(gradients)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, FLAGS.max_gradient_norm)
        self.param_norm = tf.global_norm(params)

        # Define optimizer and updates
        # (updates is what you need to fetch in session.run to do a gradient update)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        opt = tf.train.AdadeltaOptimizer(learning_rate=FLAGS.learning_rate) # you can try other optimizers
        self.updates = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

        # # Define savers (for checkpointing) and summaries (for tensorboard)
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        self.bestmodel_saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        self.summaries = tf.summary.merge_all()

        return

    def add_placeholders(self):
        self.inputs = tf.placeholder(tf.float32, (None, self.width, self.height, 3))
        self.labels = tf.sparse_placeholder(tf.int32)
        return

    def build_graph(self):
        # self.inputs = tf.transpose(self.inputs, [0,2,1,3])
        with tf.variable_scope('conv_net'):
            ConvNet = ConvolutionalNet()
            conv_output = ConvNet.build_graph(self.inputs)
            mprint('conv_output', conv_output.get_shape())
            mprint('self.inputs', self.inputs.get_shape())

        with tf.variable_scope('map2seq'):
            Map2Seq = MapToSequence()
            mapped_output = Map2Seq.build_graph(conv_output)
            mprint('mapped_output', mapped_output.get_shape())


        # with tf.variable_scope('rnn'):
        #     RNN = RNNLayer(self.hidden_size, self.alphabet_size)
        #     seq_output = RNN.build_graph(mapped_output)

        seq_len = [self.FLAGS.max_char] * tf.shape(self.inputs)[0]
        with tf.variable_scope('rnn_1'):
            RNN = RNNLayer(self.hidden_size)
            rnn_1 = RNN.build_graph(mapped_output, seq_len)
            mprint('rnn_1 output', rnn_1.get_shape())            


        with tf.variable_scope('rnn_2'):
            RNN = RNNLayer(self.alphabet_size+1)
            output = RNN.build_graph(rnn_1, seq_len)
            mprint('output', output.get_shape())            
            # RNN = RNNLayer(self.hidden_size)
            # rnn_2 = RNN.build_graph(rnn_1, self.seq_len)

        # with tf.variable_scope('affine'):
        #     affine = tf.layers.dense(rnn_2, units=self.alphabet_size+1)
        #     output = tf.reshape(affine, [self.FLAGS.batch_size, -1, self.alphabet_size+1]) # Dims: (N x ? x 52)

        with tf.variable_scope('ctc_beam_search'):
            seq_output = tf.transpose(output, perm=[1,0,2]) # Dim: (W x N X 52)
            self.model_output = seq_output
            decoded, logits = tf.nn.ctc_beam_search_decoder(seq_output, [self.FLAGS.max_char] * tf.shape(self.inputs)[0])
            self.decoded = decoded[0]
            self.logits = logits
            mprint('conv_output', self.decoded.get_shape())
        return

    def add_ctc_loss(self):
        self.loss = tf.reduce_mean(tf.nn.ctc_loss(labels=self.labels, inputs=self.model_output, sequence_length=[self.FLAGS.max_char] * tf.shape(self.inputs)[0]))
        self.edit_dist = tf.reduce_mean(tf.edit_distance(tf.cast(self.decoded, tf.int32), self.labels))
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('edit_dist', self.edit_dist)
        return


    def train(self, data_dir):
        train_tfrecords = os.path.join(data_dir, 'train.tfrecords')
        val_tfrecords = os.path.join(data_dir, 'val.tfrecords')

        images_batch, labels_batch = input_fn(train_tfrecords, train=True, batch_size=self.FLAGS.batch_size)
        with tf.Session() as sess:
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)

            inputs, labels = sess.run([images_batch, labels_batch])
            print(inputs.shape)
            input_feed={}
            input_feed[self.inputs] = inputs
            input_feed[self.labels] = labels
            output_feed = [self.updates, self.summaries, self.loss, self.edit_dist, self.global_step, self.param_norm, self.gradient_norm]
            _,_,loss, edit_dist, global_step, param_norm, gradient_norm = sess.run(output_feed,input_feed)

        return

     
