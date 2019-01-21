import tensorflow as tf 

def conv(X,filter_size,out_channels,stride_size=1,padding='SAME',init_bias=1.0,stddev=1.0,a=None):
	X_shape=X.get_shape().as_list()
	conv_weights=tf.Variable(tf.truncated_normal([filter_size,filter_size,X_shape[-1],out_channels],dtype=tf.float32,stddev=stddev),name='weights')
	conv_biases=tf.Variable(tf.constant(init_bias,shape=[out_channels],dtype=tf.float32),name='biases')	
	conv_layer=tf.nn.conv2d(X,filter=conv_weights,strides=[1,stride_size,stride_size,1],padding=padding)
	conv_layer=tf.nn.bias_add(conv_layer,conv_biases)
	if a:
		conv_layer=a(conv_layer)
	return conv_layer

def fc(inputs, output_size, init_bias=1.0, a=None, stddev=0.1):
    input_shape = inputs.get_shape().as_list()
    if len(input_shape) == 4:
        fc_weights = tf.Variable(
            tf.random_normal([input_shape[1] * input_shape[2] * input_shape[3], output_size], dtype=tf.float32,
                             stddev=stddev),
            name='weights')
        inputs = tf.reshape(inputs, [-1, fc_weights.get_shape().as_list()[0]])
    else:
        fc_weights = tf.Variable(tf.random_normal([input_shape[-1], output_size], dtype=tf.float32, stddev=stddev),
                                 name='weights')

    fc_biases = tf.Variable(tf.constant(init_bias, shape=[output_size], dtype=tf.float32), name='biases')
    fc_layer = tf.matmul(inputs, fc_weights)
    fc_layer = tf.nn.bias_add(fc_layer, fc_biases)
    if a:
        fc_layer = a(fc_layer)
    return fc_layer

def lrn(X, depth_radius=2, alpha=0.0001, beta=0.75, bias=1.0):
    return tf.nn.local_response_normalization(X, depth_radius=depth_radius, alpha=alpha, beta=beta, bias=bias)


def batch_norm(X,momentum=0.99,epsilon=0.001,training=False):
	return tf.layers.batch_normalization(X,momentum=momentum,epsilon=epsilon,training=training,name='Batch_Norm')

def maxpool(X,filter_size=3,stride_size=2,padding='VALID'):
	return tf.nn.max_pool(X,ksize=[1,filter_size,filter_size,1],strides=[1,stride_size,stride_size,1],padding=padding,name='MaxPool')

def avgpool(X,filter_size=3,stride_size=2,padding='VALID'):
	return tf.nn.avg_pool(X,ksize=[1,filter_size,filter_size,1],strides=[1,stride_size,stride_size,1],padding=padding,name='AvgPool')


















