import numpy as np

# This is specific to the pong environment
def img_crop(img):
    return img[:, 30:-12,:,:]

# GENERAL Atari preprocessing steps
def downsample(img):
    # We will take only half of the image resolution
    return img[:, ::2, ::2,:]

def transform_reward(reward):
    return np.sign(reward)

def to_grayscale(img):
    return np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])

# Normalize grayscale image from -1 to 1.
def normalize_grayscale(img):
    return (img - 128) / 128 - 1  

def process_frame(img):
    img = np.array(img)
    img = img_crop(img)
    img = downsample(img)    # Crop and downsize (by 2)
    img = to_grayscale(img)       # Convert to greyscale by averaging the RGB values
    img = normalize_grayscale(img)  # Normalize from -1 to 1.

    return np.expand_dims(img, axis=0)
