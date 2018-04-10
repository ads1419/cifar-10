import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from data import get_data_set
from model import model

test_x, test_y, test_l = get_data_set("test", cifar=10)
x, y, output, global_step, y_pred_cls = model()

image_size = 32
num_channels = 3
batch_size = 128
class_size = 10
save_path = "./tensorboard/cifar-10/"

def plot_images(images, cls_true, cls_pred=None, smooth=True):
    assert len(images) == len(cls_true) == 9
    fig, axes = plt.subplots(3, 3)

    if cls_pred is None:
        hspace = 0.3
    else:
        hspace = 0.6
    fig.subplots_adjust(hspace=hspace, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        if smooth:
            interpolation = 'spline16'
        else:
            interpolation = 'nearest'

        ax.imshow(images[i, :, :, :],
                  interpolation=interpolation)
            
        cls_true_name = test_l[np.argmax(cls_true[i], axis=0)]

        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true_name)
        else:
            cls_pred_name = test_l[cls_pred[i]]

            xlabel = "True: {0}\nPred: {1}".format(cls_true_name, cls_pred_name)

        ax.set_xlabel(xlabel)   
        
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()

images = test_x[0:9]
images = np.reshape(images, [-1, image_size, image_size, num_channels])
cls_true = test_y[0:9]
plot_images(images=images, cls_true=cls_true, smooth=True)

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

i = 0
predicted_class = np.zeros(shape=len(test_x), dtype=np.int)
while i < len(test_x):
    j = min(i + batch_size, len(test_x))
    batch_xs = test_x[i:j, :]
    batch_ys = test_y[i:j, :]
    predicted_class[i:j] = sess.run(y_pred_cls, feed_dict={x: batch_xs, y: batch_ys})
    i = j


images = test_x[10:19]
images = np.reshape(images, [-1, image_size, image_size, num_channels])
cls_true = test_y[10:19]
plot_images(images=images, cls_true=cls_true, cls_pred=predicted_class[10:19], smooth=True)

images = test_x[100:109]
images = np.reshape(images, [-1, image_size, image_size, num_channels])
cls_true = test_y[100:109]
plot_images(images=images, cls_true=cls_true, cls_pred=predicted_class[100:109], smooth=True)

images = test_x[200:209]
images = np.reshape(images, [-1, image_size, image_size, num_channels])
cls_true = test_y[200:209]
plot_images(images=images, cls_true=cls_true, cls_pred=predicted_class[200:209], smooth=True)

images = test_x[300:309]
images = np.reshape(images, [-1, image_size, image_size, num_channels])
cls_true = test_y[300:309]
plot_images(images=images, cls_true=cls_true, cls_pred=predicted_class[300:309], smooth=True)

correct = (np.argmax(test_y, axis=1) == predicted_class)
acc = correct.mean()*100
correct_numbers = correct.sum()
print("\n\nAccuracy on Test-Set: {0:.2f}% ({1} / {2})".format(acc, correct_numbers, len(test_x)))


sess.close()
