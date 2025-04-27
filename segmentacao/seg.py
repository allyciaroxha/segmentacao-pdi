import cv2
import numpy as np
from matplotlib import pyplot as plt

image_path = 'Segmenta/Outra/vaca.jpg'
image = cv2.imread(image_path)

hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower_green = np.array([36, 25, 25])
upper_green = np.array([86, 255, 255])

lower_blue = np.array([85, 77, 1])
upper_blue = np.array([200, 255, 255])

mask_green = cv2.inRange(hsv_image, lower_green, upper_green)
mask_blue = cv2.inRange(hsv_image, lower_blue, upper_blue)

mask_combined = cv2.bitwise_or(mask_green, mask_blue)

mask_inverted = cv2.bitwise_not(mask_combined)

result = cv2.bitwise_and(image, image, mask=mask_inverted)

gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

_, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

max_contour = max(contours, key=cv2.contourArea)

mask_object = np.zeros_like(binary)
cv2.drawContours(mask_object, [max_contour], -1, 255, thickness=cv2.FILLED)

object_extracted = cv2.bitwise_and(result, result, mask=mask_object)

object_gray = cv2.cvtColor(object_extracted, cv2.COLOR_BGR2GRAY)

_, object_binary = cv2.threshold(object_gray, 1, 255, cv2.THRESH_BINARY)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))

object_binary_closed = cv2.morphologyEx(object_binary, cv2.MORPH_CLOSE, kernel, iterations=2)

cv2.imwrite("objeto_extratido_binario.jpg", object_binary_closed)

result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
object_extracted_rgb = cv2.cvtColor(object_extracted, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.title("Imagem Segmentada")
plt.imshow(result_rgb)

plt.subplot(1, 3, 2)
plt.title("Objeto Extraído")
plt.imshow(object_extracted_rgb)

plt.subplot(1, 3, 3)
plt.title("Objeto Binarizado com Fechamento")
plt.imshow(object_binary_closed, cmap='gray')

plt.show()