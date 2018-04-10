import numpy as np
import tensorflow as tf

from data import get_data_set
from model import model

train_x, train_y, train_l = get_data_set("train")
test_x, test_y, test_l = get_data_set("test")

x, y, output, global_step, y_pred_cls = model()

image_size = 32
num_channels = 3
batch_size = 128
class_size = 10
iteration = 150000
save_path = "./tensorboard/cifar-10/"


loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-3).minimize(loss, global_step=global_step)


correct_prediction = tf.equal(y_pred_cls, tf.argmax(y, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


saver = tf.train.Saver()
sess = tf.Session()


try:
    print("Trying to restore last checkpoint ...")
    last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=save_path)
    saver.restore(sess, save_path=last_chk_path)
    print("Restored checkpoint from:", last_chk_path)
except:
    print("Failed to restore checkpoint. Initializing variables instead.")
    sess.run(tf.global_variables_initializer())


def train(num_iterations):

    for i in range(num_iterations):
        index = np.random.randint(len(train_x), size=batch_size)
        batch_xs = train_x[index]
        batch_ys = train_y[index]

        i_global, _ = sess.run([global_step, optimizer], feed_dict={x: batch_xs, y: batch_ys})

        if (i_global % 10 == 0) or (i == num_iterations - 1):
            _loss, batch_acc = sess.run([loss, accuracy], feed_dict={x: batch_xs, y: batch_ys})
            msg = "Step Count: {0:>6}, accuracy: {1:>6.1%}, loss = {2:.2f}"
            print(msg.format(i_global, batch_acc, _loss))

        if (i_global % 100 == 0) or (i == num_iterations - 1):
            global_1 = sess.run([global_step], feed_dict={x: batch_xs, y: batch_ys})
            acc = predict_test()

            saver.save(sess, save_path=save_path, global_step=global_step)
            print("Saved checkpoint.")


def predict_test(show_confusion_matrix=False):

    i = 0
    predicted_class = np.zeros(shape=len(test_x), dtype=np.int)
    while i < len(test_x):
        j = min(i + batch_size, len(test_x))
        batch_xs = test_x[i:j, :]
        batch_ys = test_y[i:j, :]
        predicted_class[i:j] = sess.run(y_pred_cls, feed_dict={x: batch_xs, y: batch_ys})
        i = j

    correct = (np.argmax(test_y, axis=1) == predicted_class)
    acc = correct.mean()*100
    correct_numbers = correct.sum()
    print("Accuracy on Test-Set: {0:.2f}% ({1} / {2})".format(acc, correct_numbers, len(test_x)))

    return acc


if iteration != 0:
    train(iteration)


sess.close()
