#Import All the Necessary libraries
import glob
from math import pi
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

class Canny_Edge_Detection():
  def __init__(self,
               image,
               sigma,
               kernel_size,
               high_th_ratio_for_canny,
               low_th_ratio_for_canny):
    """Init function

    Args:
        image: Numpy array or Image array
        sigma: int/float
          standard deviation of Gaussian kernel
        kernel_size: odd integer
          number of datapoints in the in the kernel
        high_threshold_for_canny= int or float
          ratio of high threshold and maximum intensity of the image
        low_threshold_for_canny = int or float
          ratio of low threshold and maximum intensity of the image
    """
    self.image=image
    self.sigma=sigma
    self.kernel_size=kernel_size
    self.high_threshold_ratio=high_th_ratio_for_canny
    self.low_threshold_ratio=low_th_ratio_for_canny

  def get_one_dimensional_Gaussian_mask(self, kernel_size, sigma):
    ##### If kernel size is even then handle by this way(add 1)
    if kernel_size%2==0:
      kernel_size=kernel_size+1
    ### determine how many points from the center of kernel
    data_points = kernel_size//2
    #### initialize kernel and derivative of the kernel with zeroes
    normal = 1 / (2.0 * np.pi * sigma**2)
    kernel = np.zeros([kernel_size,1])
    kernel_der = np.zeros([kernel_size,1])

    #### Now replace all the zero values in kernel with gaussian values
    for i in range(-data_points, data_points+1):
      kernel[i+data_points,:]=np.exp(-(i*i)/(2*sigma*sigma))*normal
      kernel_der[i+data_points,:]=kernel[i+data_points,:] * (-i)/(sigma*sigma)
    #return the kernel and derivative kernel values
    #print(kernel.T, kernel_der.T)
    return kernel.T, kernel_der.T

  def apply_conv(self, image, kernel):
      '''
      Applies convolution operation on each patch of an image using the 1D kernel.

      Parameters
      ----------
      image: Numpy Array
          The image array
      kernel: Numpy Array
          The kernel
      '''
      #print(image)

      # Flip the kernel array in x and y axis
      kernel_flipped = np.flip(kernel, axis=(0, 1))
      # Calculate the number of required padding in each direction of the image
      # to keep the output shape similar as the input image
      padding_x = kernel.shape[1] // 2
      padding_y = kernel.shape[0] // 2
      # Initiate the padded image
      image_padded = np.zeros([image.shape[0] + (padding_y*2),
                              image.shape[1] + (padding_x*2)])
      # Replace the zero pixels at the center with the original image pixel
      if padding_x == 0:
          image_padded[padding_y:-padding_y, :] = image
      elif padding_y == 0:
          image_padded[:, padding_x:-padding_x] = image
      else:
          image_padded[padding_y:-padding_y, padding_x:-padding_x] = image

      # Initiate the output image with zero elements.
      output = np.zeros([image.shape[0], image.shape[1]])

      # Handle kernel orientation in x and y axis
      if padding_x == 0:
          for y in range(image_padded.shape[0] - kernel.shape[0] + (kernel.shape[0] % 2)):
              for x in range(image_padded.shape[1] - kernel.shape[1] + (kernel.shape[1] % 2)):
                  # Isolate an image patch with the same shape as the kernel
                  image_patch = image_padded[y:(y+kernel.shape[0]), [x]]
                  # Replace the output element with the convolution result
                  output[y, x] = np.sum(image_patch * kernel_flipped)

      elif padding_y == 0:
          for y in range(image_padded.shape[0] - kernel.shape[0] + (kernel.shape[0] % 2)):
              for x in range(image_padded.shape[1] - kernel.shape[1] + (kernel.shape[1] % 2)):
                  # Isolate an image patch with the same shape as the kernel
                  image_patch = image_padded[[y], x:(x+kernel.shape[1])]
                  # Replace the output element with the convolution result
                  output[y, x] = np.sum(image_patch * kernel_flipped)

      # Return the convolved image
      #print(output)
      return output




  def Non_max_suppression(self, image, theta):


    #Apply non-max suppression to keep only the true edge pixels

    #parameters
    #image: Numpy array or Image array
    #theta: Numpy array
    #  orientation of the edge pixels from x-axis

    #initiate the non-max suppressed image with zero values
    non_max_sup_image = np.zeros(image.shape)
    # bound the theta value from 0 to pi
    theta = theta * 180. / np.pi
    theta[theta<0] += pi


    # Check each element in the image gradient magnitude array.
    for r in range(1, image.shape[0]-1):
        for c in range(1, image.shape[1]-1):
            # Initiate the two pixels along both directions of the orientation
            pixel_1 = 255
            pixel_2 = 255

            # Check the orientation angle within each pi/4 angular distance and choose two pixels accordingly
            if (0 <= theta[r, c] < 22.5) or (157.5 <= theta[r, c] <= 180):
                pixel_1 = image[r, c+1]
                pixel_2 = image[r, c-1]
            elif (22.5 <= theta[r, c] < 67.5):
                pixel_1 = image[r+1, c+1]
                pixel_2 = image[r-1, c-1]
            elif (67.5 <= theta[r, c] < 112.5):
                pixel_1 = image[r+1, c]
                pixel_2 = image[r-1, c]
            elif (112.5 <= theta[r, c] < 157.5):
                pixel_1 = image[r+1, c-1]
                pixel_2 = image[r-1, c+1]

            # If the center pixel is not smaller than both of the pixels, then it is kept as such
            if (image[r, c] >= pixel_1) and (image[r, c] >= pixel_2):
                non_max_sup_image[r, c] = image[r, c]
            # If the center pixel is smaller than either of the pixels, it is suppressed to zero
            else:
                non_max_sup_image[r, c] = 0

    # Return the non-max suppressed image
    #print(non_max_sup_image)
    return non_max_sup_image



  def apply_hysteresis_thresholding(self, image, high_threshold_ratio, low_threshold_ratio):

      # Calculate high and low threshold for hysteresis from the ratio values
      high_threshold = image.max() * high_threshold_ratio
      low_threshold = image.max() * low_threshold_ratio

      # Initiate the thresholded image with zero pixels
      image_th = np.zeros(image.shape)

      # Compare each pixel intensity with high and low threshold
      for r in range(1, image_th.shape[0]-1):
          for c in range(1, image_th.shape[1]-1):
              # If a pixel intensity is higher than the high threshold, it will take the maximum intensity
              if image[r, c] > high_threshold:
                  image_th[r, c] = 255
              # If a pixel intensity is lower than the low threshold, it will take the zero intensity
              elif image[r, c] < low_threshold:
                  image_th[r, c] = 0
              # If a pixel intensity is in between, neighbor pixels are considered
              elif low_threshold <= image[r, c] <= high_threshold:
                  # If any of the neighboring pixel intensity is higher than the high threshold,
                  # the pixel will take maximum intensity (edge); otherwise will be suppressed to zero.
                  if (image[r+1, c+1] > high_threshold) or \
                  (image[r+1, c] > high_threshold) or \
                  (image[r+1, c-1] > high_threshold) or \
                  (image[r-1, c+1] > high_threshold) or \
                  (image[r-1, c] > high_threshold) or \
                  (image[r-1, c-1] > high_threshold) or \
                  (image[r, c+1] > high_threshold) or \
                  (image[r, c-1] > high_threshold):
                      image_th[r, c] = 255
                  else:
                      image_th[r, c] = 0

      # Return the hysteresis thresholded image
      return image_th
  def detect_edge(self):
    #First generate gaussian in x direction and its derivative
    self.G_x, self.d_G_x = self.get_one_dimensional_Gaussian_mask(self.kernel_size, self.sigma)
    #Then generate gaussian in y direction and its derivative by transposing G_x and d_G_x
    self.G_y=self.G_x.T
    self.d_G_y=self.d_G_x.T
    #Apply the gaussian kernel to reduce noise in x direction
    self.Smoothed_I_x = self.apply_conv(self.image, self.G_x)
    #Apply the Gaussian kernel to reduce noise in y direction
    self.Smoothed_I_y = self.apply_conv(self.image, self.G_y)
    #apply derivative of the smoothed image in x direction
    self.I_x = self.apply_conv(self.Smoothed_I_x, self.d_G_x)
    #apply derivative of the smoothed image in y direction
    self.I_y = self.apply_conv(self.Smoothed_I_y, self.d_G_y)
    #find the magnitude of the edge response by combining the x and y components
    self.M_xy = np.sqrt(np.square(self.I_x)+np.square(self.I_y))
    ## Normalize the magnitude values
    self.M_xy=(self.M_xy/self.M_xy.max())*255
    #print(self.M_xy)
    #Calculate the orientation of of the dderivative magnitude
    self.theta = np.arctan2(self.I_y, self.I_x)
    #print(self.theta)
    ##########
    self.non_max_sup_image_I = self.Non_max_suppression(self.M_xy, self.theta)
    #plt.imshow(self.non_max_sup_image_I, cmap='gray')
    self.I_edge = self.apply_hysteresis_thresholding(self.non_max_sup_image_I,self.high_threshold_ratio,self.low_threshold_ratio)

  def plot_result(self):
      '''
      Plots the Canny edges along with intermediate results like smoothing in x and y direction, derivative in x and y directions, magnitude, non-mamimum supression result, and canny edges
      '''
      fig, ax = plt.subplots(4, 2, figsize=(15, 15))

      plt.subplot(421)
      plt.imshow(self.image, cmap='gray')
      plt.title('(a) Original Image')
      plt.subplot(422)
      plt.imshow(self.Smoothed_I_x, cmap='gray')
      plt.title(f'(b) Horizontal Smoothing for $\sigma = $ {self.sigma} and kernel size = {self.kernel_size}')
      plt.subplot(423)
      plt.imshow(self.Smoothed_I_y, cmap='gray')
      plt.title(f'(c) Vertical Smoothing for $\sigma = $ {self.sigma} and kernel size = {self.kernel_size}')
      plt.subplot(424)
      plt.imshow(self.I_x, cmap='gray')
      plt.title(f'(d) Horizontal Edges for $\sigma = $ {self.sigma} and kernel size = {self.kernel_size}')
      plt.subplot(425)
      plt.imshow(self.I_y, cmap='gray')
      plt.title(f'(e) Vertical Edges for $\sigma = $ {self.sigma} and kernel size = {self.kernel_size}')
      plt.subplot(426)
      plt.imshow(self.non_max_sup_image_I, cmap='gray')
      plt.title(f'(f) Non-maximum suppression image for $\sigma = $ {self.sigma} and kernel size = {self.kernel_size}')
      plt.subplot(427)
      plt.imshow(self.M_xy, cmap='gray')
      plt.title(f'(g) Magnitude Image for $\sigma = $ {self.sigma} and kernel size = {self.kernel_size}')
      plt.subplot(428)
      plt.imshow(self.I_edge, cmap='gray')
      plt.title(f'(h) Canny Edge Image for $\sigma = $ {self.sigma} and kernel size = {self.kernel_size}')





