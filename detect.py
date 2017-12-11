import pickle
import numpy as np
from data_processor import *
import tensorflow as tf
from tool import *
from model import LeNet

def detect():
    ### Replace each question mark with the appropriate value.
    ### Use python, pandas or numpy methods rather than hard coding the results

    batch_size = 32

    train_dataset = ImageDataset(datasets="data/train.p",
                                 transform=[
                                     lambda x: normalize(x),
                                     # lambda x: augment(x),
                                 ],
                                 )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
    )


    valid_dataset = ImageDataset(datasets="data/valid.p",
                                 transform=[
                                     lambda x: normalize(x),
                                     # lambda x: augment(x),
                                 ],
                                 )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=8,
    )


    # TODO: Number of training examples
    n_train = len(train_dataset)

    # TODO: Number of validation examples
    n_validation = len(valid_dataset)

    # TODO: Number of testing examples.
    # n_test = len(test_dataset)

    # TODO: What's the shape of an traffic sign image?
    image_shape = (train_dataset.width, train_dataset.height)

    # TODO: How many unique classes/labels there are in the dataset.
    n_classes = len(np.unique(train_dataset.labels))


    print("Number of training examples =", n_train)
    # print("Number of testing examples =", n_test)
    print("Image data shape =", image_shape)
    print("Number of classes =", n_classes)

    rate = 0.001

    #  ===== Tensor flow predifined =====

    img_tensor     = tf.placeholder(tf.float32, (None, 32, 32, 3))
    label_tensor   = tf.placeholder(tf.int32, (None))
    one_hot_tensor = tf.one_hot(label_tensor, n_classes)

    logits = LeNet(img_tensor)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_tensor, logits=logits)
    loss_operation = tf.reduce_mean(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_tensor, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    optimizer = tf.train.AdamOptimizer(learning_rate=rate)
    training_operation = optimizer.minimize(loss_operation)
    saver = tf.train.Saver()
    # ====================================


    epochs = 20

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        print("Training...")

        for epoch in range(epochs):

            for iter, (images, labels, indices) in enumerate(train_loader, 0):
                images = np.stack(img for img in images)


                labels = np.array(labels)
                sess.run(training_operation, feed_dict={img_tensor: images, label_tensor: labels})

            total_accuracy = 0
            sess = tf.get_default_session()

            for iter, (images, labels, indices) in enumerate(valid_loader, 0):
                images = np.stack(img for img in images)
                accuracy = sess.run(accuracy_operation, feed_dict={img_tensor: images, label_tensor: labels})
                total_accuracy += (accuracy * len(labels))

            validation_accuracy = total_accuracy / n_validation
            print("EPOCH {} ...".format(epoch + 1))
            print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        saver.save(sess, './lenet')

def test():


    test_dataset = ImageDataset(datasets="data/test.p",
                                transform=[
                                    lambda x: normalize(x),
                                    # lambda x: augment(x),
                                ],
                                )
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
    )

    # Remove the previous weights and bias
    tf.reset_default_graph()

    img_tensor = tf.placeholder(tf.float32, (None, 32, 32, 3))
    label_tensor = tf.placeholder(tf.int32, (None))
    one_hot_tensor = tf.one_hot(label_tensor, 43)

    logits = LeNet(img_tensor)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_tensor, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver()

    test_accuracy = 0
    n_validation = len(test_dataset)

    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('.'))

        for iter, (images, labels, indices) in enumerate(test_loader, 0):
            images = np.stack(img for img in images)
            accuracy = sess.run(accuracy_operation, feed_dict={img_tensor: images, label_tensor: labels})
            test_accuracy += (accuracy * len(labels))

        validation_accuracy = test_accuracy / n_validation
        print("Test Accuracy = {:.3f}".format(validation_accuracy))


if __name__ == '__main__':
    # detect()
    test()
    pass