import tensorflow as tf 
import ops as op


def placeholders(img_size,img_channel,label_cnt):
	with tf.name_scope('input'):
		X=tf.placeholder(shape=[None,img_size,img_size,img_channel],dtype=tf.float32,name='image')
		y=tf.placeholder(shape=[None,label_cnt],dtype=tf.float32,name='target')

	with tf.name_scope('hparams'):
		learning_rate=tf.placeholder(shape=None,dtype=tf.float32,name='learning_rate')
		dropout_keep_prob=tf.placeholder(shape=None,dtype=tf.float32,name='keep_prob')

	return X,y,learning_rate,dropout_keep_prob


def network(X,label_cnt,dropout_keep_prob=0.8):   
	with tf.name_scope('conv1layer'):
		X=op.conv(X,filter_size=11,out_channels=96,stride_size=4,padding='VALID',a=tf.nn.relu)
		X=tf.nn.max_pool(X,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID')
		X=op.lrn(X)

	with tf.name_scope('conv2layer'):
		X=op.conv(X,filter_size=5,out_channels=256,stride_size=1,padding='VALID',a=tf.nn.relu)
		X=tf.nn.max_pool(X,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID')
		X=op.lrn(X)

	with tf.name_scope('conv3layer'):
		X=op.conv(X,filter_size=3,out_channels=384,stride_size=1,padding='SAME',a=tf.nn.relu)
		
	with tf.name_scope('conv4layer'):	
		X=op.conv(X,filter_size=3,out_channels=384,stride_size=1,padding='SAME',a=tf.nn.relu)
		
	with tf.name_scope('conv5layer'):
		X=op.conv(X,filter_size=3,out_channels=256,stride_size=1,padding='SAME',a=tf.nn.relu)
		X=tf.nn.max_pool(X,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID')

	with tf.name_scope('fc1layer'):
		X=op.fc(X,output_size=9216,a=tf.nn.relu)
	with tf.name_scope('fc2layer'):	
		X=op.fc(X,output_size=4096,a=tf.nn.relu)
	with tf.name_scope('fc3layer'):
		X=op.fc(X,output_size=label_cnt,a=None)
	
	with tf.name_scope('softmaxlayer'):
		out_probs=tf.nn.softmax(logits=X,axis=-1,name='softmax_op')
		
		return X,out_probs


def loss(logits,labels):
	with tf.name_scope('loss'):
		loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=labels))
	tf.summary.scalar('loss',loss)
	return loss

def accuracy(logits,labels):
	with tf.name_scope('accuracy'):
		accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1)), tf.float32))
	tf.summary.scalar('accuracy', accuracy)
	return accuracy

def optimizer(loss,learning_rate):
	with tf.name_scope('AdamOptimizer'):
		optimizer = tf.train.AdamOptimizer(learning_rate)
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			train_op = optimizer.minimize(loss)
	return train_op