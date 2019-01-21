# take_input module

import numpy as np 
import os
from scipy.misc import imread, imresize


def image_net_data(dirname='ILSVRC',num_classes=10,num_images=20,shuffle=True):
	dst=os.path.join(os.getcwd(),dirname)
	name_dirs=os.listdir(dst)[:num_classes]
	if not os.path.isdir(os.path.join(os.getcwd(),'data')):
		os.mkdir(os.path.join(os.getcwd(),'data'))
	i=0
	for x in name_dirs:
		file_name=x+'_data.npy'
		label_file=x+'_label.npy'
		print('dirs :: ',x)
		a=preprocess_image_batch(os.path.join(dst,x),num_images=num_images)
		print('processing for class {} done'.format(x))
		
		l=[]
		for k in range(a.shape[0]):
			l.append(x)
		l=np.reshape(l,[-1,1])

		np.save(os.getcwd(),'data',file_name,a)
		np.save(os.getcwd(),'data',label_file,np.asarray(a))
		if i>0:
			final_file=np.vstack([final_file,a])
			labels=np.vstack([labels,l])
		else:
			final_file=a
			labels=l
		i+=1
	file_name=os.path.join(os.getcwd(),'data','image_net.npy')
	label_file=os.path.join(os.getcwd(),'data','image_net_label.npy')
	if shuffle:
		indices=np.arange(final_file.shape[0])
		np.random.shuffle(indices)
		final_file=final_file[indices]
		labels=labels[indices]
	labels=dense_to_one_hot(labels)
	np.save(file_name,final_file)
	np.save(label_file,labels)
	print('final_file shape :: {}'.format(final_file.shape))
	return final_file,labels
	

def preprocess_image_batch(image_paths, img_size=(256, 256), crop_size=(224, 224),num_images=20):
	i=0
	x=0
	imgs=os.listdir(image_paths)[:num_images]
	for img_name in imgs:
		im_path=os.path.join(os.getcwd(),image_paths,img_name)
		try:
			img = imread(im_path, mode='RGB')
		except:
			print('INVALID IMAGE {}'.format(im_path))
			x=1
		if x==0:
			img = imresize(img,img_size)
			img = img.astype('float32')
			# We normalize the colors (in RGB space) with the empirical means on the training set
			img[:, :, 0] -= 123.68
			img[:, :, 1] -= 116.779
			img[:, :, 2] -= 103.939
			img = img[(img_size[0] - crop_size[0]) // 2:(img_size[0] + crop_size[0]) // 2, (img_size[1]-crop_size[1])//2:(img_size[1]+crop_size[1])//2, :];
			if i==0:
				img_ret=img
			else:
				img_ret=np.vstack((img_ret,img))
			i=i+1
		x=0
	return np.reshape(img_ret,[-1,224,224,3])


def input_data(num_classes,num_images,val_num,test_num):
	DATA_ADDR=os.path.join(os.getcwd(),'data','image_net.npy')
	LABEL_ADDR=os.path.join(os.getcwd(),'data','image_net_label.npy')
	if os.path.isfile(DATA_ADDR)==0:
		print('Data not present in npy file ....')
		print('Reading and making a numpy file ....')
		image_npy,labels=image_net_data('ILSVRC',num_classes=num_classes,num_images=num_images)
	else:
		print('DATA and LABELS found Retrieving .... ')
		image_npy=np.load(DATA_ADDR)
		labels=np.load(LABEL_ADDR)

	train_num=int(np.floor((1-val_num-test_num)*image_npy.shape[0]))
	test_num=int(np.floor((1-val_num)*image_npy.shape[0]))

	indices=np.arange(image_npy.shape[0])
	np.random.shuffle(indices)
	indices_train=indices[:train_num]
	indices_test=indices[train_num:test_num]
	indices_val=indices[test_num:]

	train_images,train_labels=image_npy[indices_train],labels[indices_train]
	test_images,test_labels=image_npy[indices_test],labels[indices_test]
	val_images,val_labels=image_npy[indices_val],labels[indices_val]

	return train_images,train_labels,test_images,test_labels,val_images,val_labels

def return_batches(image_npy,labels,batch_num,batch_size):
	start=batch_num*batch_size
	end=(batch_num+1)*batch_size
	if end>image_npy.shape[0]:
		end=image_npy.shape[0]
	data=image_npy[start:end]
	labels=labels[start:end]

	return data,labels,start,end


def dense_to_one_hot(labels_dense):
	num_labels=labels_dense.shape[0]
	from sklearn.preprocessing import LabelEncoder
	le = LabelEncoder()
	labels_dense=le.fit_transform(labels_dense)
	print(labels_dense)

	from sklearn.externals import joblib
	filename = 'labelEnc_model.joblib'
	joblib.dump(le, filename)
	
	num_classes=np.shape(np.unique(labels_dense))[0]
	index_offset=np.arange(num_labels)*num_classes
	labels_one_hot=np.zeros((num_labels,num_classes))
	labels_one_hot.flat[index_offset+labels_dense.ravel()]=1
	return labels_one_hot







'''
def generate_random_integers(_sum, n):  
    mean = _sum / n
    variance = int(0.25 * mean)

    min_v = mean - variance
    max_v = mean + variance
    array = [min_v] * n

    diff = _sum - min_v * n
    while diff > 0:
        a = random.randint(0, n - 1)
        if array[a] >= max_v:
            continue
        array[a] += 1
        diff -= 1
    print array

 '''