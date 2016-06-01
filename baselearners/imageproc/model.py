import copy
import cv2
from matplotlib import pyplot as plt
import numpy as np
import regional
from scipy import interpolate
from skimage import segmentation
from skimage.draw import ellipse
from skimage.feature import peak_local_max

class ImageProcModel():
  
  # Amount to increase around a neuron when extracting a template
  TEMPLATE_BUFFER = 5
  
  def __init__(self):
    pass
  
  def _prep_images(self, images, visualize=False):
    
    blur = lambda img, sigma: cv2.GaussianBlur(img, (0, 0), sigma, None, sigma, cv2.BORDER_DEFAULT)
    
    global_mean = np.mean(images, axis=0).astype(np.float32)
    blurred_global_mean = blur(global_mean, 10)
    
    preped_images = np.subtract(images.astype(np.float32), blurred_global_mean)
    
    if visualize:    
      plt.figure(figsize=(10,10))
      plt.subplot(1, 2, 1)
      plt.imshow(global_mean)
      plt.axis('off')
      plt.title("Global Mean Image")
      plt.subplot(1, 2, 2)
      plt.imshow(blurred_global_mean)
      plt.axis('off')
      plt.title("Blurred Global Mean Image")

      plt.figure(figsize=(10,10))
      plt.subplot(1,2,1)
      plt.imshow(images[0])
      plt.axis('off')
      plt.title("Original Frame 0")
      plt.subplot(1,2,2)
      plt.imshow(preped_images[0])
      plt.axis('off')
      plt.title("Background Subtracted Frame 0")
    
    return preped_images
    
  def _extract_template(self, region, images, make_square=True, visualize=False):
    """
    Extract a template for this region
    """
    
    image_dims = images[0].shape
    
    binary_mask = np.zeros(image_dims)
    binary_mask[zip(*region.coordinates)] = 1
    
    center_y, center_x = region.center
    min_y, min_x, max_y, max_x = region.bbox

    min_y = max(0, min_y - self.TEMPLATE_BUFFER)
    max_y = min(image_dims[0], max_y + self.TEMPLATE_BUFFER)
    min_x = max(0, min_x - self.TEMPLATE_BUFFER)
    max_x = min(image_dims[1], max_x + self.TEMPLATE_BUFFER)

    # it is nice to have square templates. 
    if make_square:
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
      
    else:
      x1 = min_x
      x2 = max_x
      y1 = min_y
      y2 = max_y

    neuron_templates = images[:, y1:y2, x1:x2]
    template = np.mean(neuron_templates, axis=0)
    
    if visualize:
      
      neuron_mask = binary_mask[y1:y2, x1:x2]
    
      activations = []
      for template in neuron_templates:
        masked_template = neuron_mask * template
        activations.append(np.mean(masked_template))
    
      high_activations_indices = np.argsort(activations).tolist()
      high_activations_indices.reverse()
      
      plt.figure()
      plt.subplot(1,3,1)
      plt.imshow(np.mean(neuron_templates, axis=0))
      plt.axis('off')
      plt.title("Mean Neuron Template")

      plt.subplot(1,3,2)
      plt.imshow(neuron_mask)
      plt.axis('off')
      plt.title("Neuron Mask")
      
      plt.subplot(1,3,3)
      plt.imshow(np.mean(neuron_templates[high_activations_indices[:20]], axis=0))
      plt.axis('off')
      plt.title("Mean Neuron Template from top 20 Activations")

      plt.figure()
      plt.plot(np.arange(len(activations)), activations)
      plt.title("Activations per frame")

      # show the frames with the "highest activations" 
      plt.figure(figsize=(15, 15))
      for i, frame_index in enumerate(high_activations_indices[:5]):
        plt.subplot(1, 5, i+1)
        plt.imshow(neuron_templates[frame_index])
        plt.axis('off')
        plt.title("Frame %d" % (frame_index,))
  
    # What do we want to return as the actual template? 
    return template
  
  def train(self, images, regions, visualize=False):
    
    # compute a background image and subtract it from the frames. 
    preped_images = self._prep_images(images, visualize=visualize)
    
    self.regions = regions#copy.deepcopy(regions)
    for region in self.regions:
      region.template = self._extract_template(region, preped_images, make_square=True, visualize=visualize)
  
  def add_to_training_set(self):
    pass  
  
  def _compute_correlation(self, images, template):
    xcorrs = [cv2.matchTemplate(image, template, cv2.TM_CCORR_NORMED) for image in images]
    # max, mean, median? 
    xcorr = np.max(np.array(xcorrs), axis=0)
    
    # scale the correlation to be between 0 and 1
    f = interpolate.interp1d([-1, 1], [0, 1])
    
    xcorr_img = np.zeros(images[0].shape)
    y1 = template.shape[0]/2
    y2 = y1 + xcorr.shape[0]
    x1 = template.shape[1]/2
    x2 = x1 + xcorr.shape[1]
    xcorr_img[y1:y2,x1:x2] = f(xcorr)
    
    return xcorr_img
  
  def _non_max_suppression(self, xcorr_img, threshold_abs=0.65, min_distance=10):
    return peak_local_max(xcorr_img, threshold_abs=threshold_abs, min_distance=min_distance)
  
  
  def _construct_region_for_location(self, location, images, template, mean_image):
    """
    Given a location, build a mask around it. Just do some simple segmenting
    to construct the mask.
    """
  
    image_dims = images[0].shape
    template_dims = template.shape
    template_area = template_dims[0] * template_dims[1]
    
    template_half_height = template_dims[0] / 2
    template_half_width = template_dims[1] / 2
    y, x = location
    min_x = max(0, x - template_half_width)
    max_x = min(image_dims[1], x + template_half_width)
    min_y = max(0, y - template_half_height)
    max_y = min(image_dims[0], y + template_half_height)
    
    neuron = mean_image[min_y:max_y, min_x:max_x]
    neuron_area = neuron.shape[0] * neuron.shape[1]
    
    f = interpolate.interp1d([np.min(neuron), np.max(neuron)], [-1, 1])
    scaled_neuron = f(neuron)
    seg = segmentation.felzenszwalb(scaled_neuron, scale=1000, sigma=1, min_size=int(.4 * template_area))

    # Is this mask good to go? 
    seg_g2g = True

    # if the mask is more than 75% percent of the window, its probably bad
    if np.sum(seg) >= 0.75 * neuron_area:
      seg_g2g = False

    # did we even find a region? 
    if np.count_nonzero(seg) == 0:
      seg_g2g = False

    # did we find too many regions? 
    if np.unique(seg).shape[0] != 2: # we only want 0s and 1s
      seg_g2g = False

    # we didn't find a good segmentation
    if not seg_g2g:      
      # Just make an ellipse? 
      # Rather than an ellipse, we should just use the template
      cx = neuron.shape[1] / 2
      cy = neuron.shape[0] / 2
      rx = cx * 3 / 4
      ry = cy * 3 / 4
      seg = np.zeros(neuron.shape)
      rr, cc = ellipse(cy, cx, ry, rx)
      seg[rr, cc] = 1
  
    mask = np.zeros(image_dims)
    mask[min_y:max_y, min_x:max_x] = seg
    
    coordinates = np.transpose(np.nonzero(mask)).tolist()
    
    return regional.one(coordinates)
    
  def test(self, images):
    """
    Return neurons
    
    Are the images already processed? 
    """
    
    preped_images = self._prep_images(images, visualize=False)
    
    mean_image = np.mean(preped_images, axis=0)
    
    new_regions = []
    
    for region in self.regions:
      
      template = region.template
      
      xcorr_img = self._compute_correlation(preped_images, template)
      peaks = self._non_max_suppression(xcorr_img, threshold_abs=0.7, min_distance=template.shape[0] / 2)
      
      for location in peaks:
        correlation = xcorr_img[location]
        new_region = self._construct_region_for_location(location, preped_images, region.template, mean_image)
        new_region.conf = correlation
        new_regions.append(new_region)
    
    return new_regions
    
  def update(self):
    pass
  