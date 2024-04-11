import os
import cv2
import time

class Yolo4DNN:

    def __init__(self, nms_threshold, conf_threshold, class_labels, image_path, path_to_config, path_to_weights):      

        self.nms_threshold = nms_threshold
        self.conf_threshold = conf_threshold
        self.class_labels = class_labels
        self.image_path = image_path
        self.path_to_config = path_to_config
        self.path_to_weights = path_to_weights

        with open(class_labels, 'r') as read_class:
            classes = [label.strip() for label in read_class.readlines()]


    def load_images(self, image_path):

        img_list = []

        for img_original in os.listdir(img_path):
            if img_original endswith('.jpg') or img_original endswith('.jpeg') or img_original endswith('.png'):

                img_full_path = os.path.join(image_path, img_original)
                img_list.append(img_full_path)
            
            return img_list