"""
Utility functions for loading the datasets, working with the ROIs, and
showing images. 
"""
from glob import glob
import json
from matplotlib import pyplot as plt
import neurofinder
import numpy as np
import os
import regional
from scipy import interpolate
from scipy.misc import imread

def load_images(dataset_path):
  """
  Load in the images for this dataset
  """
  
  files = sorted(glob(os.path.join(dataset_path, 'images/*.tiff')))
  imgs = np.array([imread(f) for f in files])
  return imgs

def load_regions(dataset_path):
  """
  Load in the ROIs for a dataset.
  Returns a regional.many object
  """
  with open(os.path.join(dataset_path, 'regions/regions.json')) as f:
    data = json.load(f)
  
  regions = []
  for i in range(len(data)):
    regions.append(regional.one(data[i]['coordinates']))
  
  return regional.many(regions)

def plot_neuron_mask(regions, image_dims):
  plt.figure(figsize=(10,10))
  plt.imshow(regions.mask(dims=image_dims))
  #plt.axis('off')
  #plt.autoscale(tight=True)

def plot_neuron_boundaries(regions, image_dims):
  plt.figure(figsize=(10,10))
  plt.imshow(regions.mask(dims=image_dims, stroke='red', fill=None), cmap='gray')
  #plt.axis('off')
  #plt.autoscale(tight=True)
  
def plot_neuron_locations(regions, image_dims, new_figure=True):
  
  centers = np.array([r.center for r in regions])
  
  if new_figure:
    plt.figure(figsize=(10,10))
  plt.ylim(image_dims[0], 0)
  plt.xlim(0, image_dims[1])
  plt.plot(centers[:,1], centers[:,0], 'ro')
  #plt.autoscale(tight=True)

def plot_neuron_boundaries_on_image(image, regions):
  
  boundaries = regions.mask(dims=image.shape, stroke='red', fill=None, base=np.zeros(image.shape))
  boundaries = (boundaries[:,:,0] > 0) + 0
  
  mx = np.max(image)
  mn = np.min(image)
  
  f = interpolate.interp1d([mn, mx], [0, 255])
  g_i_r = f(image).astype(np.uint8)
  g_i_r[boundaries > 0] = 255
  g_i_g = f(image).astype(np.uint8)
  g_i_g[boundaries > 0] = 0
  g_i_b = f(image).astype(np.uint8)
  g_i_b[boundaries > 0] = 0
  i = np.dstack([g_i_r, g_i_g, g_i_b])
  
  plt.figure(figsize=(10,10))
  plt.imshow(i)

def dedup_neurons(old, new):
  """
  We want to intelligently merge in the new neurons
  """
  pass

def compute_accuracy(gt, predictions, threshold=np.inf):
  """
  compare predicted rois vs ground truth rois
  """
  a = neurofinder.load(json.dumps([{'coordinates' : r.coordinates.tolist()} for r in gt]))
  b = neurofinder.load(json.dumps([{'coordinates' : r.coordinates.tolist()} for r in predictions]))

  recall, precision = neurofinder.centers(a, b, threshold=threshold)
  inclusion, exclusion = neurofinder.shapes(a, b, threshold=threshold)

  if recall == 0 and precision == 0:
    combined = 0
  else:
    combined = 2 * (recall * precision) / (recall + precision)

  #result = {'combined': round(combined, 4), 'inclusion': round(inclusion, 4), 'precision': round(precision, 4), 'recall': round(recall, 4), 'exclusion': round(exclusion, 4)}
  #print(json.dumps(result))
  
  print "Combined: %f" % (combined,)
  print "Precision: %f" % (precision,)
  print "Recall: %f" % (recall,)
  print "Inclusion: %f" % (inclusion,)
  print "Exclusion: %f" % (exclusion,) 
  
