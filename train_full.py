# Trains network on the full neurofinder dataset

## IMPORT

import json
import numpy as np
from scipy.misc import imread
from glob import glob
import os
import random
import neurofinder as nf

from keras.layers import Input, Convolution2D, MaxPooling2D, Convolution3D, MaxPooling3D, merge, Dropout, BatchNormalization, Reshape, Dense
from keras.optimizers import SGD
from keras.models import Model


###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################

## UTILITY FUNCTIONS

def load_images(im_dir):
    # load the images
    files = sorted(glob(os.path.join(im_dir,'*.tiff')))
    imgs = np.array([imread(f) for f in files])
    return imgs

def load_regions(reg_dir):
    # load the regions (training data only)
    with open(os.path.join(reg_dir,'regions.json')) as f:
        regions = json.load(f)
    return regions

def tomask(coords,dims):
    # creates a mask from region labels
    mask = np.zeros(dims)
    mask[zip(*coords)] = 1
    return mask

def get_mask(imgs,regions):
    # creates the total mask
    dims = imgs.shape[1:]
    masks = np.array([tomask(s['coordinates'],dims) for s in regions])
    total_mask = masks.sum(axis=0)
    indices = total_mask >= 1.0
    total_mask[indices] = 1.0
    return total_mask

def subtract_background(imgs):
    # subtracts off the background from the images
    global_mean = np.mean(imgs, axis=0)
    global_mean = global_mean.astype(np.float32)
    preped_images = np.subtract(imgs.astype(np.float32), global_mean)
    return preped_images


###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################

## PATCH FUNCTIONS

def create_patches(imgs,patch_size,stride=[1,1,1],dataset_num=0):
    # enumerate coordinates for all patches of size patch_size using stride
    
    total_frames,total_height,total_width = imgs.shape
    patch_frames,patch_height,patch_width = patch_size
    f_stride,h_stride,w_stride = stride
    
    num_patches = len(xrange(0,total_height-patch_height+1,h_stride))*len(xrange(0,total_width-patch_width+1,w_stride))*len(xrange(0,total_frames-patch_frames+1,f_stride))
    
    patch_coords = np.zeros((num_patches,4))
    patch_num = 0
    
    for h in range(0,total_height-patch_height+1,h_stride):
        for w in range(0,total_width-patch_width+1,w_stride):
            for f in range(0,total_frames-patch_frames+1,f_stride):
                patch_coords[patch_num] = [dataset_num,f,h,w]
                patch_num += 1

    return patch_coords



def split_patches(patch_coords,mask,patch_size):
    # splits the patch coordinates into neuron and non-neuron patches
    
    num_patches = patch_coords.shape[0]
    patch_frames,patch_height,patch_width = patch_size
    
    neuron_patches = []
    non_neuron_patches = []
    
    for patch_num in range(num_patches):
        dataset_num,f,h,w = [int(coord) for coord in patch_coords[patch_num]]
        if mask[h+(patch_height/2),w+(patch_width/2)] == 1.0:
            neuron_patches.append(patch_num)
        else:
            non_neuron_patches.append(patch_num)

    neuron_patch_coords = patch_coords[neuron_patches]
    non_neuron_patch_coords = patch_coords[non_neuron_patches]
    
    return neuron_patch_coords,non_neuron_patch_coords



def extract_patches(coords,total_imgs,patch_size):
    # extract patch(es) of size patch_size from imgs at location coords
    patch_frames,patch_height,patch_width = patch_size
    
    num_patches = coords.shape[0]
    img_patches = np.zeros((num_patches,1,patch_frames,patch_height,patch_width))
    
    for patch_num in range(num_patches):
        dataset_num,f,h,w = [int(coord) for coord in coords[patch_num]]
        imgs = total_imgs[dataset_num]
        img_patches[patch_num,0,:,:,:] = imgs[f:f+patch_frames,h:h+patch_height,w:w+patch_width]
    
    return img_patches


###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################

## MODEL

def create_model(input_size=[2900,15,15]):
    t,h,w = input_size
    input_sequence = Input(shape=(1,t,h,w)) # (channels,frames,height,width)
    
    # conv1: spatial convolution (3 x 3), spatial pooling (2 x 2)
    conv_1 = Convolution3D(50,1,3,3,activation='relu',border_mode='same')
    conv1 = conv_1(input_sequence)
    bn1 = BatchNormalization(axis=1)(conv1)
    
    pool_1 = MaxPooling3D(pool_size=(1,2,2),strides=(1,2,2))
    pool1 = pool_1(bn1) # output size: t, h/2, w/2
    
    
    # conv2: temporal convolution (4), temporal pooling (2)
    conv_2 = Convolution3D(50,5,1,1,activation='relu',border_mode='same')
    conv2 = conv_2(pool1)
    bn2 = BatchNormalization(axis=1)(conv2)
    
    pool_2 = MaxPooling3D(pool_size=(2,1,1),strides=(2,1,1))
    pool2 = pool_2(bn2) # output size: t/2, h/2, w/2
    
    drop3 = Dropout(0.5)(pool2)
    
    # conv3: spatial convolution (3 x 3), spatial pooling (2 x 2)
    conv_3 = Convolution3D(50,1,3,3,activation='relu',border_mode='same')
    conv3 = conv_3(drop3)
    bn3 = BatchNormalization(axis=1)(conv3)
    
    pool_3 = MaxPooling3D(pool_size=(1,2,2),strides=(1,2,2))
    pool3 = pool_3(bn3) # output size: t/2, h/4, w/4
    
    
    # conv4: temporal convolution (4), temporal pooling (2)
    conv_4 = Convolution3D(50,4,1,1,activation='relu',border_mode='same')
    conv4 = conv_4(pool3)
    bn4 = BatchNormalization(axis=1)(conv4)
    
    pool_4 = MaxPooling3D(pool_size=(2,1,1),strides=(2,1,1))
    pool4 = pool_4(bn4) # output size: t/4, h/4, w/4
    
    pool_5 = MaxPooling3D(pool_size=(t/4,1,1),strides=(t/4,1,1))
    pool5 = pool_5(pool4) # output size: 1, h/4, w/4
    
    drop5 = Dropout(0.5)(pool5)
    
    # fully connected layers
    reshape6 = Reshape((50*(h/4)*(w/4),))(drop5)
    fc_6 = Dense(1000,activation = 'relu')
    fc6 = fc_6(reshape6)
    
    fc_7 = Dense(2,activation='softmax')
    fc7 = fc_7(fc6)
    
    model = Model(input=input_sequence,output=fc7)
    sgd = SGD(lr=0.1,decay=1e-6,momentum=0.9,nesterov=True)
    model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])
    
    return model






def shuffle_data(patch_coords):
    
    num_patches = patch_coords.shape[0]
    order = range(num_patches)
    random.shuffle(order)
    new_patch_coords = patch_coords[order]
    
    return new_patch_coords





def train_model(model,train_patch_coords,val_patch_coords,train_imgs,val_imgs,patch_size,batch_size=1,val_batch_size=500,display_int=20,val_int=200):
    
    # separate sets for neurons/non-neurons
    neuron_train,non_neuron_train = train_patch_coords
    neuron_val,non_neuron_val = val_patch_coords
    
    num_neuron_train = neuron_train.shape[0]
    num_non_neuron_train = non_neuron_train.shape[0]
    
    num_neuron_val = neuron_val.shape[0]
    num_non_neuron_val = non_neuron_val.shape[0]
    
    # indices for neuron/non-neuron training/validation sets
    neuron_train_it = 0
    non_neuron_train_it = 0
    neuron_val_it = 0
    non_neuron_val_it = 0
    
    num_batches = 0
    display_loss = [] # hold the losses over display_int iterations
    
    
    while True:
        
        # create the batch
        batch_coords = np.zeros((batch_size,4))
        batch_labels = np.zeros((batch_size,2))
        
        for i in range(batch_size):
            if random.random() > 0.5: # pull in a neuron
                batch_coords[i,:] = neuron_train[neuron_train_it:neuron_train_it+1]
                batch_labels[i,1] = 1.0
                neuron_train_it = (neuron_train_it+1) % num_neuron_train
            else: # pull in a non-neuron
                batch_coords[i,:] = non_neuron_train[non_neuron_train_it:non_neuron_train_it+1]
                batch_labels[i,0] = 1.0
                non_neuron_train_it = (non_neuron_train_it+1) % num_non_neuron_train
        
        batch_examples = extract_patches(batch_coords,train_imgs,patch_size)
        
        # train the model on the batch
        loss = model.train_on_batch(batch_examples,batch_labels)
        display_loss.append(loss)
        
        # display the loss
        if num_batches % display_int == 0:
            print 'Iteration: ' + str(num_batches) + ', Loss: ' + str(np.mean(display_loss)) # show average loss since last display
            display_loss = []
        
        # validation
        if num_batches % val_int == 0:
            
            neuron_acc = 0
            non_neuron_acc = 0
            num_neurons = 0
            
            for i in range(val_batch_size):
            
                if random.random() > 0.5: # neuron example
                    coords = neuron_val[neuron_val_it:neuron_val_it+1]
                    example = extract_patches(coords,val_imgs,patch_size)
                    prediction = model.predict(example,batch_size=1)
                    if prediction[0,1] > prediction[0,0]: # correct prediction
                        neuron_acc += 1
                    num_neurons += 1
                    neuron_val_it = (neuron_val_it+1) % num_neuron_val
                else: # non-neuron example
                    coords = non_neuron_val[non_neuron_val_it:non_neuron_val_it+1]
                    example = extract_patches(coords,val_imgs,patch_size)
                    prediction = model.predict(example,batch_size=1)
                    if prediction[0,0] > prediction[0,1]: # correct prediction
                        non_neuron_acc += 1
                    non_neuron_val_it = (non_neuron_val_it+1) % num_non_neuron_val
                    
            overall_acc = (neuron_acc + non_neuron_acc)*1.0/val_batch_size
            neuron_acc /= 1.0*num_neurons
            non_neuron_acc /= 1.0*(val_batch_size - num_neurons)
            
            print 'Iteration: ' + str(num_batches) + ', Overall Accuracy: ' + str(overall_acc) + ', Neuron Accuracy: ' + str(neuron_acc) + ', Non-Neuron Accuracy: ' + str(non_neuron_acc)
        
        num_batches += 1
    
    return model






###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################

# PARAMETERS

PATCH_SIZE = [512,15,15]
PATCH_STRIDE = [100,3,3]

###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################

## LOAD THE DATA

#training_sets = ['00.00','00.01','00.03','00.04','00.05','00.06','00.07','00.09','00.10','00.11','01.00','01.01','02.01','03.00']

training_sets = ['00.00']

training_images = []
training_labels = []

print 'Loading Training Data'

for dataset in training_sets:
    print 'Dataset ' + dataset
    
    images = load_images(os.path.join('data/neurofinder.' + dataset,'images'))
    preped_images = subtract_background(images)
    
    regions = load_regions(os.path.join('data/neurofinder.' + dataset,'regions'))
    mask = get_mask(images,regions)
    
    training_images.append(preped_images)
    training_labels.append(mask)

print '...done'


val_sets = ['00.02']

val_images = []
val_labels = []

print 'Loading Validation Data'

for dataset in val_sets:
    print 'Dataset ' + dataset
    
    images = load_images(os.path.join('data/neurofinder.' + dataset,'images'))
    preped_images = subtract_background(images)
    
    regions = load_regions(os.path.join('data/neurofinder.' + dataset,'regions'))
    mask = get_mask(images,regions)
    
    val_images.append(images)
    val_labels.append(mask)

print '...done'

'''
testing_sets = ['00.00.test','00.01.test','01.00.test','01.01.test','02.00.test','02.01.test','03.00.test']

testing_images = {}

print 'Loading Testing Data'

for dataset in testing_sets:
    print 'Dataset ' + dataset
    
    images = load_images(os.path.join('data/neurofinder.' + dataset,'images'))
    preped_images = subtract_background(images)
    
    testing_images.append(images)

print '...done'
'''

###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################

## CREATE PATCHES


training_neuron_patches = []
training_non_neuron_patches = []

print 'Creating Training Patches'

for dataset in training_sets:
    print 'Dataset ' + dataset
    
    dataset_num = training_sets.index(dataset)
    
    images = training_images[dataset_num]
    mask = training_labels[dataset_num]
    
    patch_coords = create_patches(images,PATCH_SIZE,PATCH_STRIDE,dataset_num) # create the patch coordinates
    neuron_patch_coords,non_neuron_patch_coords = split_patches(patch_coords,mask,PATCH_SIZE) # split them according to label
    neuron_patch_coords = shuffle_data(neuron_patch_coords)
    non_neuron_patch_coords = shuffle_data(non_neuron_patch_coords)

    if training_neuron_patches == []:
        training_neuron_patches = neuron_patch_coords
        training_non_neuron_patches = non_neuron_patch_coords
    else:
        training_neuron_patches = np.append(training_neuron_patches,neuron_patch_coords,axis=0)
        training_non_neuron_patches = np.append(training_non_neuron_patches,non_neuron_patch_coords,axis=0)

training_patches = [training_neuron_patches,training_non_neuron_patches]

print '...done'


val_neuron_patches = []
val_non_neuron_patches = []

print 'Creating Validation Patches'

for dataset in val_sets:
    print 'Dataset ' + dataset
    
    dataset_num = val_sets.index(dataset)
    
    images = val_images[dataset_num]
    mask = val_labels[dataset_num]
    
    patch_coords = create_patches(images,PATCH_SIZE,PATCH_STRIDE,dataset_num) # create the patch coordinates
    neuron_patch_coords,non_neuron_patch_coords = split_patches(patch_coords,mask,PATCH_SIZE) # split them according to label
    neuron_patch_coords = shuffle_data(neuron_patch_coords)
    non_neuron_patch_coords = shuffle_data(non_neuron_patch_coords)
    
    if val_neuron_patches == []:
        val_neuron_patches = neuron_patch_coords
        val_non_neuron_patches = non_neuron_patch_coords
    else:
        val_neuron_patches = np.append(val_neuron_patches,neuron_patch_coords,axis=0)
        val_non_neuron_patches = np.append(val_non_neuron_patches,non_neuron_patch_coords,axis=0)

val_patches = [val_neuron_patches,val_non_neuron_patches]

print '...done'
'''

testing_patches = []

print 'Creating Testing Patches'

for dataset in testing_sets:
    print 'Dataset ' + dataset
    
    dataset_num = testing_sets.index(dataset)
    
    images = testing_images[dataset_num]
    
    patch_coords = create_patches(images,PATCH_SIZE,PATCH_STRIDE,dataset_num) # create the patch coordinates
    
    if testing_patches == []:
        testing_patches = patch_coords
    else:
        testing_patches = np.append(testing_patches,patch_coords,axis=0)

print '...done'

'''


###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################

print 'Creating model'
model = create_model(PATCH_SIZE)

print 'Training model'
model = train_model(model,training_patches,val_patches,training_images,val_images,PATCH_SIZE,batch_size=5)























