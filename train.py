import tensorflow as tf
from dataset_helper import *
from modules import *
import numpy as np
from data_provider import TFRecordReader
from data_utils import *
import time
import sys

# Define a few constants
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('input', "train.tfrecords", "Path to the training file including filename")
flags.DEFINE_boolean('gpu', False, 'Indicate wheter to run on GPU')

flags.DEFINE_integer("batch_size", 32, "Defaults to 32")
flags.DEFINE_integer("train_steps", 40000, "Defaults to 40000")
flags.DEFINE_integer("print_every", 1, "Defaults to 1")
flags.DEFINE_integer("save_every", 500, "Defaults to 500")
flags.DEFINE_integer("lr_decay_steps", 10000, "Defaults to 10000")

flags.DEFINE_float("lr_decay_rate", 0.001, "Defaults to 0.1")
flags.DEFINE_float("lr", 5e-4, "Defaults to 5e-4")

flags.DEFINE_string("exp_name", "default", "Experiment name. Used to save summaries.")


def get_tboard_path():
    curr_path = os.path.dirname(os.path.abspath(__file__))  
    exp_path = os.path.join(curr_path, 'experiments')
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    exp_path = os.path.join(exp_path, FLAGS.exp_name)
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    return exp_path

def initialize_model(sess, saver):
    exp_path = get_tboard_path()
    exp_path = os.path.join(exp_path, 'ckpts')
    print ("Looking for model at %s..." % exp_path)

    ckpt = tf.train.get_checkpoint_state(exp_path)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""

    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)

    else:
        print ('Training model from scratch')
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
    return saver

def main(unused_args):
    if FLAGS.gpu:
        print("USE gpu")
        device = '/device:GPU:0'
    else:
        print("USE cpu")
        device = '/cpu:0'

    trainFile = FLAGS.input
    if (not os.path.exists(trainFile)):
        print("Could not find training file :", trainFile)
        exit(1)

    with tf.device(device):
        reader = TFRecordReader()
        images, anno = reader._read_feature(trainFile)

        inputs, labels = tf.train.shuffle_batch([images, anno], batch_size=32, capacity=1000+2*32, min_after_dequeue=100, num_threads=1)

        with tf.variable_scope('crnn', reuse=False):
            model_output, decoded, logits = CRNN(inputs, hidden_size=256)

        loss = tf.reduce_mean(tf.nn.ctc_loss(labels=labels, inputs=model_output, sequence_length=23 * np.ones(32), ignore_longer_outputs_than_inputs=True))

        edit_dist = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), labels))


        # Training Set-Up
        global_step = tf.Variable(0, name='global_step', trainable=False)

        # exponentially decaying learning rate
        lr_initial = FLAGS.lr
        lr  =tf.train.exponential_decay(lr_initial, global_step, FLAGS.lr_decay_steps, FLAGS.lr_decay_rate, staircase=True)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=lr).minimize(loss=loss, global_step=global_step)

        # Set up to save summaries
        tboard_path = get_tboard_path()
        print ('Saving summaries to ' + tboard_path)

        tf.summary.scalar('loss', loss)
        tf.summary.scalar('edit_dist', edit_dist)
        tf.summary.scalar('lr', lr)
        merged_summaries = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(tboard_path)

        config  = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        # Set up to save Model checkpoints
        saver = tf.train.Saver()
        bestmodel_saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        best_val_acc = None
        best_train_acc = None
        model_dir = os.path.join(tboard_path, 'ckpts')
        model_path = os.path.join(model_dir, 'model.ckpt')
        best_model_path = os.path.join(model_dir, 'best_model.ckpt')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    with tf.Session(config=config) as sess:
        saver = initialize_model(sess, saver)

        summary_writer.add_graph(sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for step in range(FLAGS.train_steps):
            tic = time.time()
            _,cost, eDist, preds, gt_labels, iter_num, summary = sess.run([optimizer, loss, edit_dist, decoded, labels, global_step, merged_summaries])
            toc = time.time()
            iter_time = toc-tic

            # compute accuracy
            preds = preds[0]
            preds = tf.sparse_to_dense(preds.indices, preds.dense_shape, preds.values, default_value=-1)
            gt_labels = tf.sparse_to_dense(gt_labels.indices, gt_labels.dense_shape, gt_labels.values, default_value=-1)
            preds, gt_labels = sess.run([preds, gt_labels])

            preds = nd_array_to_labels(preds)
            gt_labels = nd_array_to_labels(gt_labels)

            accuracy = get_accuracy(preds, gt_labels)

            if iter_num % FLAGS.print_every == 0:
                print ('Iter[{}] Loss: {:.4f}, Edit Dist: {:.4f}, Accuracy: {:.4f}, Time: {}'.format(iter_num, cost, eDist, accuracy, iter_time))
                sys.stdout.flush()


            # TODO: Evaluate of Validation Set every 1000 iterations, and save best model
            # if best_val_acc is None or best_val_acc < accuracy:
            #     best_val_acc = accuracy
            #     bestmodel_saver.save(sess, best_model_path, global_step=step)
            if best_train_acc is None or best_train_acc < accuracy:
                best_train_acc = accuracy
                bestmodel_saver.save(sess, best_model_path, global_step=step)

            summary_writer.add_summary(summary=summary, global_step=step)

            if iter_num % FLAGS.save_every == 0:
                saver.save(sess, model_path, global_step=step)            

        coord.request_stop()
        coord.join(threads)

    return

if __name__ == '__main__':
    tf.app.run()