import cv2
import numpy as np
import matplotlib.pyplot as plt


IMG_SIZE = 48


def create_example_img():
  img = np.full((IMG_SIZE, IMG_SIZE), 0, np.uint8)
  img[IMG_SIZE // 2:, :IMG_SIZE // 2] = 255
  img = cv2.GaussianBlur(img, (25, 25), 0)
  return img


def scharr_operator(img):
  kernel_x = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]) / 32
  kernel_y = np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]]) / 32
  # The function compute correlation, not the convolution.
  g_x = cv2.filter2D(img.astype(np.float32), -1, kernel_x)
  g_y = cv2.filter2D(img.astype(np.float32), -1, kernel_y)
  gradient = np.stack((g_x, g_y), axis=-1)
  return gradient


def plot_gradient(gradient, ax):
  # Change image coordinate to matplotlib coordinate.
  # We have to rotate 90 degree.
  R = np.array([[0, -1], [1, 0]])

  for i in range(IMG_SIZE):
    for j in range(IMG_SIZE):
      if (gradient[i, j, 0] ** 2 + gradient[i, j, 1] ** 2)== 0:
        # Draw dot if the length of the gradient is zero.
        ax.plot(IMG_SIZE - i, j, 'ko', markersize=1)
      else:
        g = gradient[i, j, :]
        g = R @ g
        ax.quiver(IMG_SIZE - i, j, g[0], g[1], linewidths=0.01)

  ax.set_title("Gradient")
  ax.set_aspect('equal')


if __name__ == '__main__':
  img = create_example_img()

  fig, axs = plt.subplots(1, 2, figsize=(13, 13))
  axs[0].imshow(img, cmap='gray')
  axs[0].set_title("Example Image")

  gradient = scharr_operator(img)
  plot_gradient(gradient, axs[1])

  plt.show()
