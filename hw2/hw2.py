import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity, mean_squared_error
import copy

image = cv2.imread('hw2/sar_1.jpg')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

# plt.imshow(image_gray, cmap="gray")


# Шум Гаусса

mean = 0
stddev = 100
noise_gauss = np.zeros(image_gray.shape, np.uint8)
cv2.randn(noise_gauss, mean, stddev)

# plt.imshow(noise_gauss, cmap="gray")


# постоянный шум
noise =  np.random.randint(0, 101, size = (image_gray.shape[0], image_gray.shape[1]), dtype=int)
zeros_pixel = np.where(noise == 0)
ones_pixel = np.where(noise == 100)

bg_image = np.ones(image_gray.shape, np.uint8) * 128

bg_image[zeros_pixel] = 0
bg_image[ones_pixel] = 255

# plt.imshow(bg_image, cmap="gray")


# фильтры

# медианный фильтр

image_noise_gauss = cv2.add(image_gray,noise_gauss)

mse_gauss = mean_squared_error(image_gray, image_noise_gauss)
(ssim, diff) = structural_similarity(image_gray, image_noise_gauss, full=True)
print(mse_gauss, ssim)


image_gauss_median = cv2.medianBlur(image_noise_gauss, 3)

mse_gauss_median = mean_squared_error(image_gray, image_gauss_median)
(ssim_gauss_median, diff) = structural_similarity(image_gray,  image_gauss_median, full=True)

print(mse_gauss_median, ssim_gauss_median)



# медианный
image_gauss_median = cv2.medianBlur(image_noise_gauss, 5)

# гаусса
image_gauss_gauss = cv2.GaussianBlur(image_noise_gauss,(5,5),0)

# Билатеральный
image_gauss_bilat = cv2.bilateralFilter(image_noise_gauss,9,75,75)

# Нелокальных средних
image_gauss_nlm = cv2.fastNlMeansDenoising(image_noise_gauss, h = 20)

plt.imshow(image_gauss_nlm, cmap="gray")


plt.show()