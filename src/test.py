import cv2
import matplotlib.pyplot as plt

# Replace this path with the path to an actual image file in your dataset
test_image_path = '../data/test/angry/PrivateTest_88305.jpg'

image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
if image is not None:
    print("Image successfully loaded")
    plt.imshow(image, cmap='gray')
    plt.show()
else:
    print("Failed to load image")
