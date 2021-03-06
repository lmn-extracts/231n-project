import tensorflow as tf
from dataset_helper import *
from custom_estimator import *
from modules import *
from data_utils import *
from best_exporter import *
import pickle

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
flags.DEFINE_integer("eval_throttle_secs", 600, "Defaults to 600")
flags.DEFINE_integer("eval_steps", 1000, "Defaults to 1000")
flags.DEFINE_integer("parallel_cpu", 1, "Defaults to 1")

flags.DEFINE_float("lr_decay_rate", 0.001, "Defaults to 0.1")
flags.DEFINE_float("lr", 5e-4, "Defaults to 5e-4")

flags.DEFINE_string("exp_name", "default", "Experiment name. Used to save summaries.")
flags.DEFINE_string("model_dir", None, "Path location to save Experiments.")
flags.DEFINE_string("tf_format", "JPG", "Specify whether TfRecords has image in JPG or raw format")
flags.DEFINE_string("run_type", "train_and_eval", "Run type: train_and_eval, eval_only")
flags.DEFINE_string("ckpt_path", None, "Path to the best checkpoint")

def get_tboard_path():
    if FLAGS.model_dir is None:
        base_path = os.path.dirname(os.path.abspath(__file__))
    else:
        base_path = FLAGS.model_dir
    exp_path = os.path.join(base_path, 'experiments')
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    exp_path = os.path.join(exp_path, FLAGS.exp_name)
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    return exp_path

def main(unused_args):

    trainFile = os.path.join(FLAGS.datadir, FLAGS.input)
    valFile = os.path.join(FLAGS.datadir, FLAGS.valfile)
    testFile = os.path.join(FLAGS.datadir, FLAGS.testfile)
    val_test_batch_size = FLAGS.val_batch_size

    if FLAGS.run_type == 'train_and eval' and (not os.path.exists(trainFile)):
        print("Could not find training file :", trainFile)
        exit(1)

    tboard_path = get_tboard_path()
    model_dir = os.path.join(tboard_path, 'ckpts')
    #model_path = os.path.join(model_dir, 'model.ckpt')
    #best_model_path = os.path.join(model_dir, 'best_model.ckpt')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    #feature_image = tf.feature_column.numeric_column("image",
    #                                             shape=[32, 100, 3])
    #my_feature_columns = [feature_image]

    classifier = tf.estimator.Estimator(
        model_fn=crnn_model,
        params={
            #'feature_columns': my_feature_columns,
            'hidden_size': 256,
            'batch_size': FLAGS.batch_size,
            'lr': FLAGS.lr,
            'lr_decay_steps': FLAGS.lr_decay_steps,
            'lr_decay_rate': FLAGS.lr_decay_rate,
            'test_batch_size': val_test_batch_size,
        },
        model_dir = model_dir)

    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda:input_fn(trainFile, train=True, batch_size=FLAGS.batch_size, parallel_calls=FLAGS.parallel_cpu,
                                 tf_format=FLAGS.tf_format),
        max_steps=FLAGS.train_steps)

    my_feature_columns = get_feature_columns()
    serving_feature_spec = tf.feature_column.make_parse_example_spec(my_feature_columns)
    serving_input_receiver_fn = (
        tf.estimator.export.build_parsing_serving_input_receiver_fn(
            serving_feature_spec))
    exporter = BestExporter(
        name="best_exporter",
        event_file_pattern='eval_text-eval/*.tfevents.*', # must match name in eval_spec
        serving_input_receiver_fn=serving_input_receiver_fn,
        #serving_input_receiver_fn=serving_input_fn,
        exports_to_keep=5)

    eval_spec = tf.estimator.EvalSpec(
        #input_fn=lambda:input_fn(valFile, train=False, batch_size=val_test_batch_size, parallel_calls=FLAGS.parallel_cpu,
        #                         tf_format=FLAGS.tf_format),
        input_fn=lambda: input_fn(valFile, train=True, batch_size=val_test_batch_size, parallel_calls=FLAGS.parallel_cpu,
                                 tf_format=FLAGS.tf_format),
        steps=FLAGS.eval_steps,
        exporters=exporter,
        name='text-eval',
        throttle_secs=FLAGS.eval_throttle_secs)

    if FLAGS.run_type == 'train_and_eval':
        tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)
    elif FLAGS.run_type == 'eval_only':
        if FLAGS.ckpt_path is None:
            print ('Checkpoint path --ckpt_path required.')
            return
        eval_result = classifier.evaluate(input_fn=lambda: input_fn(valFile, train=False,
                                            batch_size=val_test_batch_size, parallel_calls=FLAGS.parallel_cpu,
                                            tf_format=FLAGS.tf_format),
                                            checkpoint_path=FLAGS.ckpt_path)
        #print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result['accuracy']))

    elif FLAGS.run_type == 'test':
        if FLAGS.ckpt_path is None:
            print ('Checkpoint path --ckpt_path required.')
            return
        preds = []
        test_batch_size = 1
        for result in classifier.predict(input_fn= lambda: input_fn(testFile, train=False, batch_size=test_batch_size,
                                            parallel_calls=FLAGS.parallel_cpu,
                                            tf_format=FLAGS.tf_format),
                                        checkpoint_path = FLAGS.ckpt_path):
            pred = ''.join([idx2char(c) for c in result['predicted_label']])
            preds.append(pred)
            #print(pred)
            #print (len(preds))
        with open('predictions', 'wb') as fp:
            pickle.dump(preds, fp)


        image_batch, label_batch = input_fn(testFile, train=False, batch_size=test_batch_size, buffer_size=2048,
                                            parallel_calls=FLAGS.parallel_cpu, tf_format=FLAGS.tf_format)

        with tf.Session() as sess:
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)
            count = 0
            while True:
                try:

                    images, labels = sess.run([image_batch, label_batch])
                    dense_label = sess.run(
                        tf.sparse_to_dense(labels.indices, labels.dense_shape, labels.values, default_value=-1))
                    #print("DENSE LABEL", dense_label)
                    for i in range(test_batch_size):
                        print(count)
                        print('Prediction: ', preds[count])
                        #image_i = images['image'][i]
                        #print("DENSE LABEL i", dense_label[i])
                        label = [idx2char(c) for c in dense_label[i]]
                        #print("***Label", label)
                        print('Ground truth: ', ''.join(label))
                        #image_i = image_i.astype(np.uint8)
                        #print("image shape", image_i.shape)
                        count += 1

                        #plt.imshow(image_i)
                        #plt.title(label)
                        #plt.show()
                except tf.errors.OutOfRangeError:
                    break

    #print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result['accuracy']))


    # for i in range(3):
    #     classifier.train(
    #         input_fn=lambda:input_fn(trainFile, train=True, batch_size=FLAGS.batch_size, parallel_calls=FLAGS.parallel_cpu, tf_format=FLAGS.tf_format),
    #         steps=FLAGS.train_steps
    #     )
    #
    #     eval_result = classifier.evaluate(input_fn=lambda:input_fn(valFile,
    #                     train=False, batch_size=val_test_batch_size, parallel_calls=FLAGS.parallel_cpu, tf_format=FLAGS.tf_format),
    #                                       steps=FLAGS.eval_steps, )

    #print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result['accuracy']))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
