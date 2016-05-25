"""
Try to detect and remove camera frames that are compromised due to camera jitter.
"""
from matplotlib import pyplot as plt
import numpy as np

def remove_shakey_frames(images, visualize=False):
  """
  images needs to be in a datatype that doesn't risk rounding issues. 
  """

  # Compute the global mean from all frames
  global_mean = np.mean(images, axis=0)
  
  global_diffs = []
  for img in images:
    global_diffs.append(np.sum(np.abs(global_mean - img)))
  
  jitter_end_threshold = np.mean(global_diffs) +  1 * np.std(global_diffs)
  
  if visualize:
    plt.figure()
    plt.plot(np.arange(len(global_diffs)), global_diffs)
    plt.hlines(jitter_end_threshold, 0, len(global_diffs), label="Jitter End Threshold")
    plt.title("Difference of each frame compared to the global mean image")
    plt.legend()
  
  diff_between_frames = np.abs(np.diff(global_diffs))
  
  jitter_start_threshold = np.mean(diff_between_frames) + 10 * np.std(diff_between_frames)

  if visualize:
    plt.figure()
    plt.plot(np.arange(len(diff_between_frames)), diff_between_frames)
    plt.hlines(jitter_start_threshold, 0, len(diff_between_frames), label="Jitter Start Threshold")
    plt.title("Magnitude of difference between consecutive frames")
    plt.legend()
  
  # We want to find windows of frames that are messed up
  filtered_images = []
  i = 0
  while i < len(images) - 1:
    if diff_between_frames[i] > jitter_start_threshold:
      j = i + 1
      while j < len(images):
        if global_diffs[j] < jitter_end_threshold:
          break
        else:
          j += 1
      print "Skipping frames %d to %d" % (i, j)
      i = j
    else:
      filtered_images.append(images[i])
      i += 1
  # Don't forget about the last frame
  if i == len(images) - 1:
    filtered_images.append(images[i])
  
  return np.array(filtered_images)