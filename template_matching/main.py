"""Find the offset [u, v] that maximize the similarities of template and original image"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def compute_cross_correlations(template, img):
  m, n = template.shape
  max_u = img.shape[0] - m
  max_v = img.shape[1] - n
  cross_correlations = np.zeros((max_u, max_v))
  M = m * n # Number of pixels of the template.
  template_mean = np.sum(template) / M
  for u in range(max_u):
    for v in range(max_v):
      img_patch = img[u:u + m, v:v + n]
      img_patch_mean = np.sum(img_patch) / M
      accum = 0
      for i in range(m):
        for j in range(n):
          accum += (img_patch[i, j] - img_patch_mean) * (template[i, j] - template_mean)
      cross_correlations[u, v] = accum / (M - 1)

  return cross_correlations


if __name__ == '__main__':
  img = cv2.imread('resources/Unequalized_Hawkes_Bay_NZ.jpg', 0)
  img = cv2.resize(img, (360, 240))

  u, v = 104, 137 # The offset u, v
  m, n = 64, 64 # Template image size
  template = img[u:u + m, v:v + n]
  img = img * 0.9 - 30 # Take radiometric transformation into account.
  img = img.astype(np.uint8)

  fig = plt.figure(figsize=(15, 15))
  ax = fig.add_subplot(2, 2, 1)
  ax.imshow(template, vmin=0, vmax=255, cmap='gray')
  ax.set_title('Template')
  ax.axis('off')

  ax = fig.add_subplot(2, 2, 2)
  ax.imshow(img, vmin=0, vmax=255, cmap='gray')
  ax.set_title('Image')
  ax.axis('off')

  cross_correlations = compute_cross_correlations(template, img)
  u_hat, v_hat = np.unravel_index(cross_correlations.argmax(), cross_correlations.shape)

  matched_img = cv2.rectangle(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), (u_hat, v_hat),
      (u_hat + m, v_hat + n), (0, 0, 255), 2)
  ax = fig.add_subplot(2, 2, 3)
  ax.imshow(matched_img[:,:,::-1])
  ax.set_title('Matched Image')
  ax.axis('off')

  x, y = np.meshgrid(np.arange(cross_correlations.shape[1]), np.arange(cross_correlations.shape[0]))
  ax = fig.add_subplot(2, 2, 4, projection='3d')
  surf = ax.plot_surface(x, y, cross_correlations, cmap=cm.coolwarm, linewidth=0, antialiased=False)
  fig.colorbar(surf, shrink=0.3, aspect=7)
  ax.set_title('Cross Correlations')

  plt.show()
