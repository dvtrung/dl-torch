import tensorflow.compat.v1 as tf


def relu(inp, relu_type: str = "elu", **kwargs):
    relu_type = relu_type.lower()
    if relu_type == "prm":
        with tf.variable_scope(None, default_name="prelu"):
            alpha = tf.get_variable(
                "alpha", shape=inp.get_shape()[-1],
                initializer=tf.constant_initializer(0.25))
            pos = tf.nn.relu(inp)
            neg = - (alpha * tf.nn.relu(-inp))
            output = pos + neg
    elif relu_type == "elu":
        output = tf.nn.elu(inp)
    elif relu_type == "lky":
        assert 'alpha' in kwargs
        output = tf.maximum(inp, kwargs['alpha'] * inp)
    elif relu_type == "std":  # STD
        output = tf.nn.relu(inp)

    return output


def get_variable(scope_name, shape, variable_name=None, initializer="normal"):
    if initializer == "xavier_normal":
        initializer = tf.glorot_uniform_initializer()
    elif initializer == "normal":
        initializer = tf.random_normal_initializer()
    elif initializer == "zeros":
        initializer = tf.zeros_initializer()
    x = tf.get_variable(variable_name or scope_name, shape, initializer=initializer)
    return x


def get_activation_fn(name):
    if name is None or name == "none":
        return tf.identity
    name = name.lower()
    if name in ["tanh", "sigmoid"]:
        return getattr(tf, name)
    elif name in ["relu", "elu"]:
        return getattr(tf.nn, name)
    else:
        raise ValueError(f"{name} not supported")


def get_rnn_cell(dim: int, cell_type: str, activation_fn=None, reuse=None):
    cell_type = cell_type.lower()
    activation_fn = get_activation_fn(activation_fn)

    if cell_type == "ProjLSTM":
        cell = tf.nn.rnn_cell.LSTMCell
        if projDim is None:
            projDim = config.cellDim
        cell = cell(dim, num_proj=projDim, reuse=reuse, activation=activation_fn)
        return cell

    cells = {
        "rnn": tf.nn.rnn_cell.BasicRNNCell,
        "gru": tf.nn.rnn_cell.GRUCell,
        "lstm": tf.nn.rnn_cell.BasicLSTMCell,
        # "MiGRU": MiGRUCell,
        # "MiLSTM": MiLSTMCell
    }

    cell = cells[cell_type](dim, reuse=reuse, activation=activation_fn)

    return cell


def cnn(
        inp,
        inDim,
        outDim,
        batch_norm=None,
        dropout=0.0,
        addBias=True,
        kernel_size=None,
        stride=1,
        activation_fn=None,
        name="",
        reuse=None):
    with tf.variable_scope("cnnLayer" + name, reuse=reuse):
        kernel = get_variable("kernels", (kernel_size, kernel_size, inDim, outDim), initializer="xavier_normal")

        if batch_norm is not None:
            inp = tf.contrib.layers.batch_norm(
                inp, decay=batch_norm["decay"], center=batch_norm["center"],
                scale=batch_norm["scale"], is_training=batch_norm["train"], updates_collections=None)

        inp = tf.nn.dropout(inp, rate=dropout)
        output = tf.nn.conv2d(
            inp,
            filter=kernel,
            strides=[1, stride, stride, 1],
            padding="SAME")

        if addBias:
            b = get_variable("biases", (outDim,), initializer="zeros")
            output += b

        output = get_activation_fn(activation_fn)(output)

    return output


def multi_layer_cnn(
        x,
        dims,
        batch_norm=None,
        dropout=0.0,
        kernel_sizes=None,
        strides=None,
        activation_fn="relu"):
    if isinstance(kernel_sizes, int):
        kernel_sizes = [kernel_sizes for _ in dims]
    if isinstance(strides, int):
        strides = [strides for _ in dims]

    for i in range(len(dims) - 1):
        x = cnn(
            x,
            dims[i],
            dims[i + 1],
            name="cnn_%d" % i,
            batch_norm=batch_norm,
            dropout=dropout,
            kernel_size=kernel_sizes[i],
            stride=strides[i],
            activation_fn=activation_fn)

    return x