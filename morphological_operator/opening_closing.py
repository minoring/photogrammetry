import cv2
import numpy as np
import matplotlib.pyplot as plt


def to_binary_img(img, threshold):
  bin_img = img.copy()
  bin_img[img >= threshold] = 1
  bin_img[img < threshold] = 0

  return bin_img


def dilate(img):
  h, w = img.shape
  dilated_img = img.copy()
  for i in range(h):
    for j in range(w):
      if i - 1 >= 0:
        if j - 1 >= 0:
          dilated_img[i][j] = dilated_img[i][j] or img[i - 1][j - 1]
        if j + 1 < w:
          dilated_img[i][j] = dilated_img[i][j] or img[i - 1][j + 1]
      if i + 1 < h:
        if j - 1 >= 0:
          dilated_img[i][j] = dilated_img[i][j] or img[i + 1][j - 1]
        if j + 1 < w:
          dilated_img[i][j] = dilated_img[i][j] or img[i + 1][j + 1]
  return dilated_img


def erode(img):
  h, w = img.shape
  eroded_img = img.copy()
  for i in range(h):
    for j in range(w):
      if i - 1 >= 0:
        if j - 1 >= 0:
          eroded_img[i][j] = eroded_img[i][j] and img[i - 1][j - 1]
        if j + 1 < w:
          eroded_img[i][j] = eroded_img[i][j] and img[i - 1][j + 1]
      if i + 1 < h:
        if j - 1 >= 0:
          eroded_img[i][j] = eroded_img[i][j] and img[i + 1][j - 1]
        if j + 1 < w:
          eroded_img[i][j] = eroded_img[i][j] and img[i + 1][j + 1]
  return eroded_img


def opening(img):
  eroded_img = erode(img)
  dilated_img = dilate(eroded_img)
  return dilated_img


def closing(img):
  dilated_img = dilate(img)
  eroded_img = erode(dilated_img)
  return eroded_img


if __name__ =='__main__':
  img = cv2.imread('resources/puppet.png', 0)

  fig, axs = plt.subplots(1, 4, figsize=(16, 8))
  axs[0].imshow(img, cmap='gray')
  axs[0].set_title("Original Image")
  axs[0].axis('off')

  bin_img = to_binary_img(img, 30)
  axs[1].imshow(bin_img, cmap='gray')
  axs[1].set_title("Binary Image")
  axs[1].axis('off')


  opened_img = opening(bin_img)
  axs[2].imshow(opened_img, cmap='gray')
  axs[2].set_title("Opening Image")
  axs[2].axis('off')

  opened_closed_img = closing(opened_img)
  axs[3].imshow(opened_closed_img, cmap='gray')
  axs[3].set_title("Opening + Closing Image")
  axs[3].axis('off')

  plt.show()
