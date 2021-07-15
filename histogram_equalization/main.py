import cv2
import numpy as np
import matplotlib.pyplot as plt


def histogram_equalization(hist, N):
  cum_hist = np.cumsum(hist)
  hist_equalized = []

  for i, a in enumerate(hist):
    if a > 0:
      cdf_min = i
      break

  for a in range(256):
    hist_equalized.append(round(255 * ((cum_hist[a] - cdf_min) / (N - cdf_min))))

  hist_equalized = np.array(hist_equalized, np.uint8)
  return hist_equalized


def plot_hist(hist, bins, ax):
  """Magic function from: https://stackoverflow.com/a/5328669"""
  width = 0.7 * (bins[1] - bins[0])
  center = (bins[:-1] + bins[1:]) / 2
  ax.bar(center, hist, align='center', width=width, color='r', label='histogram')
  cdf = hist.cumsum()
  cdf_normalized = cdf * (hist.max() / cdf.max())
  ax.plot(cdf_normalized, color='b', label='cdf')
  ax.legend(loc='upper left')


if __name__ == '__main__':
  img = cv2.imread('resources/Unequalized_Hawkes_Bay_NZ.jpg', 0)
  N = img.flatten().shape[0]

  fig, axs = plt.subplots(2, 2, figsize=(8, 8))
  hist, bins = np.histogram(img.flatten(), 256, [0, 255])
  axs[0, 0].imshow(img, cmap='gray', vmin=0, vmax=255)
  axs[0, 0].axis('off')
  axs[0, 0].set_title('Original Image')
  plot_hist(hist, bins, axs[1, 0])
  axs[1, 0].set_title('Original Histogram')

  hist_equalizer = histogram_equalization(hist, N)
  img_equalized = img.copy()
  m, n = img.shape
  for i in range(m):
    for j in range(n):
      img_equalized[i, j] = hist_equalizer[img[i, j]]
  hist_equalized, bins = np.histogram(img_equalized.flatten(), 256, [0, 255])

  axs[0, 1].imshow(img_equalized, cmap='gray', vmin=0, vmax=255)
  axs[0, 1].axis('off')
  axs[0, 1].set_title('Equalized Image')
  plot_hist(hist_equalized, bins, axs[1, 1])
  axs[1, 1].set_title('Equalized Histogram')

  plt.show()
