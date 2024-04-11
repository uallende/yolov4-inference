import cv2

img_path = '/workdir/resnet18/images_detected/common newt, Triturus vulgaris.jpg'
img_path = '/workdir/resnet18/images/Carduelis_carduelis_close_up.jpg'

img_np = cv2.imread(img_path)
print(img_np)