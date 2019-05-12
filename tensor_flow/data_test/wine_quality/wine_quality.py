import tensorflow as tf
import wine_quality_data
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')

def main(argv):

    args = parser.parse_args(argv[1:])

    (train_x, train_y), (test_x, test_y) = wine_quality_data.load_data_split_by_sample()

    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    print(1)

    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        # Two hidden layers of 10 nodes each.
        hidden_units=[10, 10, 10],
        # The model must choose between 10 classes.
        n_classes=10)

    # Train the Model.
    classifier.train(
        input_fn=lambda: wine_quality_data.train_input_fn(train_x, train_y,
                                                  args.batch_size),
        steps=args.train_steps)

    # Evaluate the model.
    eval_result = classifier.evaluate(
        input_fn=lambda: wine_quality_data.eval_input_fn(test_x, test_y,
                                                 args.batch_size))

    # classifier.predict()
    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
    print(1)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
