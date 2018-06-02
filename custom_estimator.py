import tensorflow as tf
from modules import *
from data_utils import *

def crnn_model(features, labels, mode, params):

    inputs = features['image']
    print("INPUTS SHAPE", inputs.shape)

    if mode == tf.estimator.ModeKeys.TRAIN:
        batch_size = params['batch_size']
        lr_initial = params['lr']
        lr = tf.train.exponential_decay(lr_initial, global_step=tf.train.get_global_step(),
                                        decay_steps=params['lr_decay_steps'], decay_rate=params['lr_decay_rate'],
                                        staircase=True)
        tf.summary.scalar('lr', lr)
    else:
        batch_size = params['test_batch_size']

    with tf.variable_scope('crnn', reuse=False):
        rnn_output, predicted_label, logits = CRNN(inputs, hidden_size=params['hidden_size'], batch_size=batch_size)

    preds = predicted_label[0]
    preds = tf.sparse_to_dense(preds.indices, preds.dense_shape, preds.values, default_value=-1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'predicted_label': preds,
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)


    loss = tf.reduce_mean(tf.nn.ctc_loss(labels=labels, inputs=rnn_output,
                                         sequence_length=23 * np.ones(batch_size),
                                         ignore_longer_outputs_than_inputs=True))
    edit_dist = tf.reduce_mean(tf.edit_distance(tf.cast(predicted_label[0], tf.int32), labels))
    accuracy = compute_accuracy(predicted_label[0], labels)

    metrics = {
        'accuracy': tf.metrics.mean(accuracy),
        'edit_dist':tf.metrics.mean(edit_dist),
    }

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('edit_dist', edit_dist)
    tf.summary.scalar('accuracy', accuracy)

    if mode == tf.estimator.ModeKeys.EVAL:
        # eval_summ_hook = tf.train.SummarySaverHook(
        #     save_steps=200,
        #     summary_op=tf.summary.merge_all())
        #return tf.estimator.EstimatorSpec(mode, loss=loss, evaluation_hooks=[eval_summ_hook])

        #return tf.estimator.EstimatorSpec(mode, loss=loss)
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    assert mode == tf.estimator.ModeKeys.TRAIN

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0005).minimize(loss)
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=lr)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
