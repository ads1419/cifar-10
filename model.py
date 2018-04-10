import tensorflow as tf

def model():
    image_size = 32
    image_channels = 3
    num_classes = 10
    reshape_size = 2048

    with tf.name_scope("data"):
        x = tf.placeholder(tf.float32, shape=[None, image_size * image_size * image_channels], name="Input")
        y = tf.placeholder(tf.float32, shape=[None, num_classes], name="Output")
        x_image = tf.reshape(x, [-1, image_size, image_size, image_channels], name="images")

    def weight_decay_var(name, shape, stddev, wd):
        var = variable_initialize( name, shape, tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name="weight_loss")
            tf.add_to_collection("losses", weight_decay)
        return var

    def variable_initialize(name, shape, initializer):
        with tf.device("/cpu:0"):
            dtype = tf.float32
            var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
        return var

    with tf.variable_scope("conv1") as scope:
        weights = weight_decay_var("weights", shape=[5, 5, 3, 64], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(x_image, weights, [1, 1, 1, 1], padding="SAME")
        biases = variable_initialize("biases", [64], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)

    norm1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name="norm1")
    pool1 = tf.nn.max_pool(norm1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool1")

    with tf.variable_scope("conv2") as scope:
        weights = weight_decay_var("weights", shape=[5, 5, 64, 64], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(pool1, weights, [1, 1, 1, 1], padding="SAME")
        biases = variable_initialize("biases", [64], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)

    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name="norm2")
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool2")

    with tf.variable_scope("conv3") as scope:
        weights = weight_decay_var("weights", shape=[3, 3, 64, 128], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(pool2, weights, [1, 1, 1, 1], padding="SAME")
        biases = variable_initialize("biases", [128], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(pre_activation, name=scope.name)

    with tf.variable_scope("conv4") as scope:
        weights = weight_decay_var("weights", shape=[3, 3, 128, 128], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(conv3, weights, [1, 1, 1, 1], padding="SAME")
        biases = variable_initialize("biases", [128], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(pre_activation, name=scope.name)

    with tf.variable_scope("conv5") as scope:
        weights = weight_decay_var("weights", shape=[3, 3, 128, 128], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(conv4, weights, [1, 1, 1, 1], padding="SAME")
        biases = variable_initialize("biases", [128], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(pre_activation, name=scope.name)

    norm3 = tf.nn.lrn(conv5, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name="norm3")
    pool3 = tf.nn.max_pool(norm3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool3")

    with tf.variable_scope("fully_connected1") as scope:
        reshape = tf.reshape(pool3, [-1, reshape_size])
        dim = reshape.get_shape()[1].value
        weights = weight_decay_var("weights", shape=[dim, 384], stddev=0.04, wd=0.004)
        biases = variable_initialize("biases", [384], tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

    with tf.variable_scope("fully_connected2") as scope:
        weights = weight_decay_var("weights", shape=[384, 192], stddev=0.04, wd=0.004)
        biases = variable_initialize("biases", [192], tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)

    with tf.variable_scope("output") as scope:
        weights = weight_decay_var("weights", [192, num_classes], stddev=1 / 192.0, wd=0.0)
        biases = variable_initialize("biases", [num_classes], tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)

    global_step = tf.Variable(initial_value=0, name="global_step", trainable=False)
    y_pred_cls = tf.argmax(softmax_linear, axis=1)

    return x, y, softmax_linear, global_step, y_pred_cls
