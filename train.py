import tensorflow as tf
from dataset_helper import *
from custom_estimator import *
from modules import *
from data_utils import *

# Define a few constants
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('input', "train.tfrecords", "Train data filename")
flags.DEFINE_string('valfile', "val.tfrecords", "Validation data filename")
flags.DEFINE_string('testfile', "test.tfrecords", "Test data filename")
flags.DEFINE_string('datadir', './', "Path to datasets")
flags.DEFINE_boolean('gpu', False, 'Indicate wheter to run on GPU')

flags.DEFINE_integer("batch_size", 32, "Defaults to 32")
flags.DEFINE_integer("val_batch_size", 32, "Defaults to 32")
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

def main(unused_args):

    trainFile = os.path.join(FLAGS.datadir, FLAGS.input)
    valFile = os.path.join(FLAGS.datadir, FLAGS.valfile)
    #testFile = os.path.join(FLAGS.datadir, FLAGS.testfile)
    val_test_batch_size = FLAGS.val_batch_size

    if (not os.path.exists(trainFile)):
        print("Could not find training file :", trainFile)
        exit(1)

    tboard_path = get_tboard_path()
    model_dir = os.path.join(tboard_path, 'ckpts')
    #model_path = os.path.join(model_dir, 'model.ckpt')
    #best_model_path = os.path.join(model_dir, 'best_model.ckpt')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    feature_image = tf.feature_column.numeric_column("image",
                                                 shape=[32, 100, 3])

    my_feature_columns = [feature_image]

    classifier = tf.estimator.Estimator(
        model_fn=crnn_model,
        params={
            'feature_columns': my_feature_columns,
            'hidden_size': 256,
            'batch_size': FLAGS.batch_size,
            'lr': FLAGS.lr,
            'lr_decay_steps': FLAGS.lr_decay_steps,
            'lr_decay_rate': FLAGS.lr_decay_rate,
            'test_batch_size': val_test_batch_size,
        },
        model_dir = model_dir)

    classifier.train(
        input_fn=lambda:input_fn(trainFile, train=True, batch_size=FLAGS.batch_size),
        steps=FLAGS.train_steps
    )

    eval_result = classifier.evaluate(input_fn=lambda:input_fn(valFile, train=False, batch_size=val_test_batch_size))

    #print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result['accuracy']))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
