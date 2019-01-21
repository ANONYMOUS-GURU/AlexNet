import tensorflow as tf
from take_input import input_data
import AlexNet_architecture as model               ########
from take_input import input_data,return_batches
import os
import numpy as np 
import time

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('epoch', 50, "training epoch")
tf.app.flags.DEFINE_float('test_size',0.1,'test size')
tf.app.flags.DEFINE_float('val_size',0.1,'val size')
tf.app.flags.DEFINE_boolean('train',True,'training')
tf.app.flags.DEFINE_float('learning_rate_',0.001,'learning rate')
tf.app.flags.DEFINE_float('keep_prob',0.8,'keep prob')
tf.app.flags.DEFINE_string('save_name','saved_Alexnet_model','folder of saving')
tf.app.flags.DEFINE_integer('validation_interval',1000,'validation_interval')


def train():
	# take input data from take_input
	train_images,train_labels,test_images,test_labels,val_images,val_labels=input_data(num_classes=2,num_images=10,val_num=FLAGS.val_size,test_num=FLAGS.test_size)

	img_size=train_images.shape[2]
	img_channel=train_images.shape[3]
	label_cnt=train_labels.shape[1]

	# let's have the input placeholders
	X,y,learning_rate,dropout_keep_prob=model.placeholders(img_size,img_channel,label_cnt)   ####
	logits,out_probs=model.network(X,dropout_keep_prob=dropout_keep_prob,label_cnt=label_cnt)   ####
	loss=model.loss(logits,y)
	optimizer=model.optimizer(loss,learning_rate)

	init=tf.global_variables_initializer()
	sess=tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False))
	sess.run(init)

	merged=tf.summary.merge_all()
	writer_train_addr='./summary/train'
	writer_val_addr='./summary/val'
	train_writer=tf.summary.FileWriter(writer_train_addr,sess.graph)
	val_writer=tf.summary.FileWriter(writer_val_addr)

	saver=tf.train.Saver()
	saver_addr=os.path.join(os.getcwd(),FLAGS.save_name,'var.ckpt')
	if not os.path.isdir(os.path.join(os.getcwd(),FLAGS.save_name)):
		os.mkdir(FLAGS.save_name)
	if os.path.isfile(saver_addr):
		saver.restore(sess,saver_addr)

	train_num=train_images.shape[0]
	batch_size=128
	num_batches=int(np.ceil(train_num/batch_size))

	lr=FLAGS.learning_rate_
	kp=FLAGS.keep_prob
	epochs=FLAGS.epoch
	train_size=train_images.shape[0]
	is_train=FLAGS.train

	for epoch in range(epochs):
		if epoch % 3 == 0 and epoch > 0:
			lr /= 2
		i=0
		epoch_loss=0
		for batch in range(num_batches):
			train_x,train_y,start,end=return_batches(train_images,train_labels,batch_size=batch_size,batch_num=batch)
			if i%20==0:
				summary,_,batch_loss=sess.run([merged,optimizer,loss],feed_dict={X:train_x,y:train_y,learning_rate:lr,dropout_keep_prob:kp})    #########
				train_writer.add_summary(summary, i/20)
			else:
				_,_,batch_loss=sess.run([optimizer,loss],feed_dict={X:train_x,y:train_y,learning_rate:lr,dropout_keep_prob:kp})      ##########

			epoch_loss+=batch_loss
			print('>> training loss computed :: {} on {} images from {} to {} out of {}  with learning_rate {}'.format(batch_loss,train_x.shape[0],start,end,train_size,lr))

			if i%FLAGS.validation_interval==0 and i>0:
				accuracy_batch,batch_loss,summary=sess.run([accuracy,loss,summary],feed_dict={X:val_images,y:val_labels,dropout_keep_prob:1.0})        ########
				val_writer.add_summary(summary,i/FLAGS.validation_interval)

				print('>> validation loss computed :: {} and validation accuracy :: {} on {} images'.format(batch_loss,accuracy_batch,val_images.shape[0]))

		print('>> epoch loss computed :: {} '.format(epoch_loss/num_batches))
		saver.save(sess, saver_addr)
	train_writer.close()
	val_writer.close()

	sess.close()

	test(test_images,test_labels)

def test(test_images=None,test_labels=None):
	tf.reset_default_graph()
	sess=tf.Session()
	new_saver = tf.train.import_meta_graph(os.path.join(os.getcwd(),FLAGS.save_name,'var.ckpt.meta'))
	new_saver.restore(sess,os.path.join(os.getcwd(),FLAGS.save_name,'var.ckpt'))
	print('graph restored')
	ops=sess.graph.get_operations()
	#for x in ops:
	#	print(x.name)
	x=0
	try:
		a=test_images.shape
	except :
		x=1

	if x==1:
		_,_,test_images,test_labels,_,_=input_data(num_classes=2,num_images=10,val_num=FLAGS.val_size,test_num=FLAGS.test_size)

	test_size=test_images.shape[0]
	sess_input=sess.graph.get_tensor_by_name('input/image:0')
	sess_logits=sess.graph.get_tensor_by_name('fc3layer/BiasAdd:0')
	sess_keep_prob=sess.graph.get_tensor_by_name('hparams/keep_prob:0')
	#sess_training=sess.graph.get_tensor_by_name('is_train:0')                ###########

	logits_out=sess.run(sess_logits,feed_dict={sess_input:test_images,sess_keep_prob:1.0})     #######
	out=np.argmax(logits_out,1)
	print(out)
	print(test_labels)
	acc=np.sum(out==test_labels)/test_size
	print('accuracy :: {}'.format(acc))
	sess.close()

def main(_):
	if FLAGS.train:
		print('training')
		train()
	else:
		test()

if __name__=='__main__':
	tf.app.run()


















