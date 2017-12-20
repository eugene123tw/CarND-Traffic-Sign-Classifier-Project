import pickle
import numpy as np
from data_processor import *
import tensorflow as tf
from tool import *
from model import LeNet
import matplotlib.pyplot as plt


def detect():

    batch_size = 16

    train_dataset = ImageDataset(datasets="data/train.p",
                                 transform=[
                                     lambda x: randomShift(x),
                                     lambda x: randomRotate(x),
                                     lambda x: randomBrightness(x),
                                     lambda x: normalize(x),

                                 ],
                                 )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
    )

    valid_dataset = ImageDataset(datasets="data/valid.p",
                                 transform=[
                                     lambda x: normalize(x)
                                 ],
                                 )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=2,
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

    img_tensor = tf.placeholder(tf.float32, (None, 32, 32, 3))
    label_tensor = tf.placeholder(tf.int32, (None))
    one_hot_tensor = tf.one_hot(label_tensor, n_classes)
    phase = tf.placeholder(tf.bool)

    logits = LeNet(img_tensor, phase)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_tensor, logits=logits)
    loss_operation = tf.reduce_mean(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_tensor, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    ## http://ruishu.io/2016/12/27/batchnorm/
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        # Ensures that we execute the update_ops before performing the train_step
        optimizer = tf.train.AdamOptimizer(learning_rate=rate)
        training_operation = optimizer.minimize(loss_operation)


    saver = tf.train.Saver()
    # ====================================
    num_its = len(train_loader)
    epochs = 40

    ## https://github.com/tensorflow/tensorflow/issues/6698
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        print('epoch    iter   rate | train_acc   valid_acc | \n')
        print('--------------------------------------------------------------------------------------------------\n')

        for epoch in range(epochs):
            total_train_acc = 0
            for iter, (images, labels, indices) in enumerate(train_loader, 0):
                _, acc = sess.run([training_operation, accuracy_operation],
                                  feed_dict={img_tensor: images, label_tensor: labels, phase: 1})
                total_train_acc += (acc * len(labels))

            train_accuracy = total_train_acc / n_train
            validation_accuracy = do_validate(sess, valid_loader, n_validation, accuracy_operation, img_tensor, label_tensor, phase)

            print('%5.1f   %5d    %0.4f  | %0.4f  %0.4f | ... ' % \
                  (epoch + (iter + 1) / num_its, iter + 1, rate, train_accuracy, validation_accuracy))

        saver.save(sess, './lenet')


def do_validate(sess, valid_loader, n_validation,  accuracy_operation, img_tensor, label_tensor, phase):
    total_accuracy = 0

    for iter, (images, labels, indices) in enumerate(valid_loader, 0):
        accuracy = sess.run(accuracy_operation, feed_dict={img_tensor: images, label_tensor: labels, phase: 0})
        total_accuracy += (accuracy * len(labels))

    validation_accuracy = total_accuracy / n_validation
    return validation_accuracy


def test():
    batch_size = 16

    test_dataset = ImageDataset(datasets="data/test.p",
                                 transform=[
                                     lambda x: normalize(x),
                                 ],
                                 )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=batch_size,
    )


    # Remove the previous weights and bias
    tf.reset_default_graph()

    img_tensor = tf.placeholder(tf.float32, (None, 32, 32, 3))
    label_tensor = tf.placeholder(tf.int32, (None))
    one_hot_tensor = tf.one_hot(label_tensor, 43)
    phase = tf.placeholder(tf.bool)

    logits = LeNet(img_tensor, phase)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_tensor, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver()

    test_accuracy = 0
    n_validation = len(test_dataset)

    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('.'))

        for iter, (images, labels, indices) in enumerate(test_loader, 0):
            accuracy = accuracy_operation.eval(feed_dict={img_tensor: images, label_tensor: labels, phase: 0})
            test_accuracy += (accuracy * len(labels))

        validation_accuracy = test_accuracy / n_validation
        print("Test Accuracy = {:.3f}".format(validation_accuracy))


if __name__ == '__main__':
    detect()
    # test()
    pass