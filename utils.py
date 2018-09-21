import numpy as np
import os
from keras import backend as K
from keras.layers.core import Lambda
from scipy.misc.pilutil import imread
from scipy.misc import imresize
from PIL import Image

def local_response_normalization(alpha=1e-4, k=2, beta=0.75, n=5, **kwargs):
    def f(X):
        K.set_image_dim_ordering('th')
        if K.backend()=='tensorflow':
            b, ch, r, c = X.get_shape()
        else:
            b, ch, r, c = X.shape
        half = n // 2
        square = K.square(X)
        extra_channels = K.spatial_2d_padding(K.permute_dimensions(square, (0,2,3,1))
                                              , ((0,0),(half,half)))
        extra_channels = K.permute_dimensions(extra_channels, (0,3,1,2))
        scale = k
        for i in range(n):
            if K.backend()=='tensorflow':
                ch = int(ch)
            scale += alpha * extra_channels[:,i:i+ch,:,:]
        scale = scale ** beta
        return X / scale

    return Lambda(f, output_shape=lambda input_shape:input_shape,**kwargs)

def split_tensor(axis=1, ratio_split=1, id_split=0, **kwargs):
    def f(X):
        if K.backend()=='tensorflow':
            div = int(X.get_shape()[axis]) // ratio_split
        else:
            div = X.shape[axis] // ratio_split

        if axis == 0:
            output =  X[id_split*div:(id_split+1)*div,:,:,:]
        elif axis == 1:
            output =  X[:, id_split*div:(id_split+1)*div, :, :]
        elif axis == 2:
            output = X[:,:,id_split*div:(id_split+1)*div,:]
        elif axis == 3:
            output = X[:,:,:,id_split*div:(id_split+1)*div]
        else:
            raise ValueError("This axis is not possible")

        return output

    def g(input_shape):
        output_shape=list(input_shape)
        output_shape[axis] = output_shape[axis] // ratio_split
        return tuple(output_shape)

    return Lambda(f,output_shape=lambda input_shape:g(input_shape),**kwargs)

def preprocess_image_batch(image_paths, img_size=None, crop_size=None, color_mode="rgb", out=None):
    img_list = []

    for im_path in image_paths:
        img = imread(im_path, mode='RGB')
        if img_size:
            img = imresize(img,img_size)

        img = img.astype('float32')
        # We normalize the colors (in RGB space) with the empirical means on the training set
        img[:, :, 0] -= 123.68
        img[:, :, 1] -= 116.779
        img[:, :, 2] -= 103.939
        # We permute the colors to get them in the BGR order
        if color_mode=="bgr":
            img[:,:,[0,1,2]] = img[:,:,[2,1,0]]
        # img = img.transpose((2, 0, 1))
        if crop_size:
            img = img[(img_size[0]-crop_size[0])//2:(img_size[0]+crop_size[0])//2
                      ,(img_size[1]-crop_size[1])//2:(img_size[1]+crop_size[1])//2,:]

        print(img.shape)
        print('111111111111111')
        img_list.append(img)

    try:
        img_batch = np.stack(img_list, axis=0)
    except:
        raise ValueError('when img_size and crop_size are None, images'
                ' in image_paths must have the same shapes.')

    if out is not None and hasattr(out, 'append'):
        out.append(img_batch)
    else:
        return img_batch

def get_annotations_map(path):
	valAnnotationsPath = path + '/val/val_annotations.txt'
	valAnnotationsFile = open(valAnnotationsPath, 'r')
	valAnnotationsContents = valAnnotationsFile.read()
	valAnnotations = {}

	for line in valAnnotationsContents.splitlines():
		pieces = line.strip().split()
		valAnnotations[pieces[0]] = pieces[1]

	return valAnnotations

def load_images(path,num_classes):
    #Load images

    print('Loading ' + str(num_classes) + ' classes')

    X_train=np.zeros([num_classes*500,3,64,64],dtype='uint8')
    y_train=np.zeros([num_classes*500], dtype='uint8')

    trainPath=path+'/train'

    print('loading training images...');

    i=0
    j=0
    annotations={}
    for sChild in os.listdir(trainPath):
        sChildPath = os.path.join(os.path.join(trainPath,sChild),'images')
        annotations[sChild]=j
        for c in os.listdir(sChildPath):
            X=np.array(Image.open(os.path.join(sChildPath,c)))
            if len(np.shape(X))==2:
                X_train[i]=np.array([X,X,X])
            else:
                X_train[i]=np.transpose(X,(2,0,1))
            y_train[i]=j
            i+=1
        j+=1
        if (j >= num_classes):
            break

    print('finished loading training images')

    val_annotations_map = get_annotations_map(path)

    X_test = np.zeros([num_classes*50,3,64,64],dtype='uint8')
    y_test = np.zeros([num_classes*50], dtype='uint8')


    print('loading test images...')

    i = 0
    testPath=path+'/val/images'
    for sChild in os.listdir(testPath):
        if val_annotations_map[sChild] in annotations.keys():
            sChildPath = os.path.join(testPath, sChild)
            X=np.array(Image.open(sChildPath))
            if len(np.shape(X))==2:
                X_test[i]=np.array([X,X,X])
            else:
                X_test[i]=np.transpose(X,(2,0,1))
            y_test[i]=annotations[val_annotations_map[sChild]]
            i+=1
        else:
            pass


    print('finished loading test images: '+str(i))

    return X_train,y_train,X_test,y_test
