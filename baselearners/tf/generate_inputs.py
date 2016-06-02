import numpy as np
import random
from scipy.ndimage import zoom
from skimage.util import view_as_blocks
import tensorflow as tf

import sys
sys.path.append('../..')
from util import load_images, load_regions

def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
  

def write_examples(features, labels, output_path):

  writer = tf.python_io.TFRecordWriter(output_path)

  for feature, label in zip(features, labels):
    example = tf.train.Example(features=tf.train.Features(
      feature={
        'feature': _float_feature(feature),
        'label': _int64_feature([label]),
      }
    ))

    writer.write(example.SerializeToString())

  writer.close()

def generate_simple_dataset(num_samples, num_positives, input_size, num_steps, output_path):
  """
  num_samples: number of "patches"
  num_positives: number of positive examples
  input_size: size of one feature vector from one "frame"
  num_steps: number of "frames" to concatenate
  """
  
  dataset = np.zeros([num_samples, input_size * num_steps], dtype=float)
  
  positive_class_indices = random.sample(xrange(num_samples), num_positives)
  labels = np.zeros(num_samples, dtype=int)
  labels[positive_class_indices] = 1
  
  
  # Make one index of the positive classes 1
  feature_indices = range(input_size * num_steps)
  for index in positive_class_indices:
    flag_index = random.choice(feature_indices)
    dataset[index, flag_index] = 1
  
  # sanity check
  print "Number of non-zeros in dataset: %d" % (np.nonzero(dataset)[0].shape[0],)
  print "Number of non-zeros in labels: %d" % (np.nonzero(labels)[0].shape[0],)
  print "Dataset and labels make sense: %s" % (np.all(np.nonzero(dataset)[0] == np.nonzero(labels)[0]),)
  
  write_examples(dataset, labels, output_path)

def create_patch(region, images, buffer, output_shape):
    
    image_dims = images[0].shape
    
    center_y, center_x = region.center
    min_y, min_x, max_y, max_x = region.bbox

    min_y = max(0, min_y - buffer)
    max_y = min(image_dims[0], max_y + buffer)
    min_x = max(0, min_x - buffer)
    max_x = min(image_dims[1], max_x + buffer)

    # it is nice to have square templates. 
    if center_x - min_x < max_x - center_x:
      bbox_radius_w = center_x - min_x
    else:
      bbox_radius_w = max_x - center_x
  
    if center_y - min_y < max_y - center_y:
      bbox_radius_h = center_y - min_y
    else:
      bbox_radius_h = max_y - center_y
  
    if bbox_radius_w < bbox_radius_h:
      bbox_radius = bbox_radius_w
    else:
      bbox_radius = bbox_radius_h
    
    x1 = center_x - bbox_radius
    x2 = center_x + bbox_radius
    y1 = center_y - bbox_radius
    y2 = center_y + bbox_radius
    
    # Lets try to make the diameter odd while we are at it
    y2 = min(image_dims[0], y2 + 1)
    x2 = min(image_dims[1], x2 + 1)
    
    patch = images[:,y1:y2,x1:x2]
     
    scaled_patch = zoom(patch, (1,  float(output_shape[0]) / patch.shape[1] , float(output_shape[1]) / patch.shape[2]))
    
    return scaled_patch
      
def generate_dataset(neurofinder_dataset_path, patch_dims, buffer=1, max_number_of_negatives=500, output_path=None):
  """
  neurofinder_dataset_path : path to a neurofinder dataset directory
  patch_dims : (height, width) for the desired patch size
  buffer : buffer to increase the size of the patch around a neuron
  max_number_of_negatives : the maximum number of negative patches to save
  output_path : location to save the tfrecords file (or None if you don't want to save a tfrecords file)
  """
  images = load_images(neurofinder_dataset_path)
  regions = load_regions(neurofinder_dataset_path)
  
  # convert the images to floats and normalize them. 
  images = images.astype(float)
  images /= np.max(images)
  
  assert np.min(images) == 0.
  assert np.max(images) == 1.
  
  # we'll make a fixed sized patch around each neuron
  positive_patches = []
  for region in regions:
    patch = create_patch(region, images, buffer=buffer, output_shape=patch_dims)
    positive_patches.append(patch)
  
  # for the negative patches, lets consider the patches that don't contain a neuron
  # does the regional code base have a test for containment?
  negative_patches = []
  
  # create an image with 1s over the neurons and 0s else where
  masks = regions.mask(dims=images[0].shape, stroke='red', fill='red', base=np.zeros(images[0].shape))
  masks = (masks[:,:,0] > 0) + 0 
  
  video_height, video_width = images[0].shape
  patch_height, patch_width = patch_dims
  patch_area = float(patch_height * patch_width)
  h_stride, w_stride = (9, 9)
  
  for h in range(0,video_height-patch_height+1,h_stride):
    for w in range(0,video_width-patch_width+1,w_stride):
      
      p = masks[h:h+patch_height, w:w+patch_width]
      
      # make sure that neurons don't cover a significant portion of the patch
      if np.sum(p) / patch_area > .4:
        continue
      
      # make sure that a neuron is not in the middle of the patch
      if np.sum(p[patch_height / 2 - 1 : patch_height / 2 + 1, patch_width / 2 - 1 : patch_width / 2 + 1]) > 0:
        continue
      
      # Good to go
      negative_patches.append(images[:, h:h+patch_height, w:w+patch_width])
  
  print "Found %d total negative patches." % (len(negative_patches),)
  
  p_patches = [p.ravel() for p in positive_patches]
  n_patches = [p.ravel() for p in negative_patches]
  random.shuffle(n_patches)
  n_patches = n_patches[:max_number_of_negatives]
  
  
  training_set = [(p, 1) for p in p_patches] + [(p, 0) for p in n_patches]
  random.shuffle(training_set)
  
  features = [d[0] for d in training_set]
  labels = [d[1] for d in training_set]
  
  print "Number of positive patches: %d" % (len(p_patches),)
  print "Number of negatve patches: %d" % (len(n_patches),)
  print "Number of frames: %d" % (images.shape[0],)
  print "Patch Dims: %s" % (patch_dims,)
  print "Feature size: %s" % (p_patches[0].shape,)
  
  if output_path != None:
    write_examples(features, labels, output_path)
  
  return features, labels
  
  
  
    
    
  