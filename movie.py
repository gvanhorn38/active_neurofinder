"""
Create a video out of the images
"""

from matplotlib import pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy import interpolate

from glob import glob
from util import load_images, load_regions
import os
from scipy.misc import imread
from PIL import Image

def create_movie(dataset_path, output_path, outline=False, fps=30, dpi=100):
  """
  Creates a mp4 file.
  """
  
  files = sorted(glob(os.path.join(dataset_path, 'images/*.tiff')))
  images = [Image.open(files[0])]
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.set_aspect('equal')
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

  if outline: 
    regions = load_regions(dataset_path)
    boundaries = regions.mask(dims=images[0].shape, stroke='red', fill=None, base=np.zeros(images[0].shape))
    boundaries = (boundaries[:,:,0] > 0) + 0
    
    mx = np.max(images)
    mn = np.min(images)
    
    f = interpolate.interp1d([mn, mx], [0, 255])
    
    mod_images = []
    for image in images:
      g_i_r = f(image).astype(np.uint8)
      g_i_r[boundaries > 0] = 255
      g_i_g = f(image).astype(np.uint8)
      g_i_g[boundaries > 0] = 0
      g_i_b = f(image).astype(np.uint8)
      g_i_b[boundaries > 0] = 0
      i = np.dstack([g_i_r, g_i_g, g_i_b])
      mod_images.append(i)
    
    images = np.array(mod_images)
    
  im = ax.imshow(images[0] ,cmap='gray')
  
  fig.set_size_inches([5,5])
  plt.tight_layout()

  def update_img(n):
      tmp = Image.open(files[n])
      im.set_data(tmp)
      return im

  animator = animation.FuncAnimation(fig,update_img,len(files),interval=fps)
  writer = animation.writers['ffmpeg'](fps=fps)
  animator.save(output_path,writer=writer,dpi=dpi)

 
  
  
  
