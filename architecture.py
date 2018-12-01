import tensorflow as tf
import numpy as np

n_inputs=227*227*3
n_classes=1000


def fc_layer(X=None,W=None,bias=None,name=None):
    return tf.bias_add(tf.matmul(X,W),b)



def AlexNet(input_image,weights,biases):
        
        if input_image.shape==[227,227,3]:
                
                img=tf.reshape(input_image,[-1,227,227,3])
                
                conv1=tf.nn.conv2d(img,weights['wc1'],strides=[1,4,4,1],padding='SAME',name='conv1')
                conv1=tf.nn.bias_add(conv1,biases['bc1'])
                conv1=tf.nn.relu(conv1)
                conv1=tf.nn.local_response_normalization(conv1,depth_radius=5.0,bias=2.0,alpha=1e-4,beta=0.75)
                conv1=tf.max_pool(conv1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID')
                
                
                conv2=tf.nn.conv2d(conv1,weights['wc2'],strides=[1,1,1,1],padding='SAME',name='conv2')
                conv2=tf.nn.bias_add(conv2,biases['bc2'])
                conv2=tf.nn.relu(conv2)
                conv2=tf.nn.local_response_normalization(conv2,depth_radius=5.0,bias=2.0,aplha=1e-4,beta=0.75)
                conv2=tf.nn.max_pool(conv2,strides=[1,2,2,1],padding='VALID',ksize=[1,3,3,1])
                
                conv3=tf.nn.con2d(conv2,weights['wc3'],strides=[1,1,1,1],padding='SAME',name='conv3')
                conv3=tf.nn.relu(tf.nn.bias_add(conv3,biases['bc3']))
                
                conv4=tf.nn.conv2d(conv3,weights['wc4'],strides=[1,1,1,1],padding='SAME',name='conv4')
                conv4=tf.nn.relu(tf.nn.bias_add(conv4,biases['bc4']))
                
                conv5=tf.nn.conv2d(conv4,weights['wc5'],strides=[1,1,1,1],padding='SAME',name='conv5')
                conv5=tf.nn.relu(tf.nn.bias_add(conv5,biases['bc5']))
                conv5=tf.nn.max_pool(conv5,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID')
                
                
                # Flatten the last conv layer 
                flatten=tf.reshape(conv5,[-1,weights['wf1'].shape[0]])
                
                fc1=tf.nn.relu(fc_layer(flatten,weights['wf1'],biases['bf1'],'fc1'))
                fc1=tf.nn.dropout(fc2,keep_prob=0.5)
                
                fc2=tf.nn.relu(fc_layer(fc1,weights['wf2'],biases['bf2'],'fc2'))
                fc2=tf.nn.relu(fc2,keep_prob=0.5)
                
                fc3=tf.nn.softmax(fc_layer(fc2,weights['wf3'],biases['bf3'],'fc3'))
                
                
                return fc3
        else:
            print('shape mismatch')
        
        
        
weights = {
    "wc1": tf.Variable(tf.truncated_normal([11, 11, 3, 96],     stddev=0.01), name="wc1"),
    "wc2": tf.Variable(tf.truncated_normal([5, 5, 96, 256],     stddev=0.01), name="wc2"),
    "wc3": tf.Variable(tf.truncated_normal([3, 3, 256, 384],    stddev=0.01), name="wc3"),
    "wc4": tf.Variable(tf.truncated_normal([3, 3, 384, 384],    stddev=0.01), name="wc4"),
    "wc5": tf.Variable(tf.truncated_normal([3, 3, 384, 256],    stddev=0.01), name="wc5"),
    "wf1": tf.Variable(tf.truncated_normal([6*6*256, 4096],   stddev=0.01), name="wf1"),
    "wf2": tf.Variable(tf.truncated_normal([4096, 4096],        stddev=0.01), name="wf2"),
    "wf3": tf.Variable(tf.truncated_normal([4096, n_classes],   stddev=0.01), name="wf3")
}

biases = {
    "bc1": tf.Variable(tf.constant(0.0, shape=[96]),        name="bc1"),
    "bc2": tf.Variable(tf.constant(1.0, shape=[256]),       name="bc2"),
    "bc3": tf.Variable(tf.constant(0.0, shape=[384]),       name="bc3"),
    "bc4": tf.Variable(tf.constant(1.0, shape=[384]),       name="bc4"),
    "bc5": tf.Variable(tf.constant(1.0, shape=[256]),       name="bc5"),
    "bf1": tf.Variable(tf.constant(1.0, shape=[4096]),      name="bf1"),
    "bf2": tf.Variable(tf.constant(1.0, shape=[4096]),      name="bf2"),
    "bf3": tf.Variable(tf.constant(1.0, shape=[n_classes]), name="bf3")
}
