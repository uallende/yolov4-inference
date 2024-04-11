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

        self.frames = self.load_images(self.image_path)

        for self.frame in self.frames:
            self.image = cv2.imread(self.frame)
            self.origingal_h, self.original_w, _ = self.image.shape
            dimension = (640, 640)
            self.resize_img = cv2.resize(self.image, dimension, interpolation=cv2.INTER_AREA)
            self.new_h, self.new_w, _ = self.resize_img
            self.inference_run(self.resize_img)

    def load_images(self, image_path):

        img_list = []

        for img_original in os.listdir(img_path):
            if img_original.endswith('.jpg') or img_original.endswith('.jpeg') or img_original.endswith('.png'):

                img_full_path = os.path.join(image_path, img_original)
                img_list.append(img_full_path)
            
        return img_list

    def inference_dnn(self, path_to_cfg, path_to_weights):

        network = cv2.dnn.readNet(path_to_cfg, path_to_weights)
        network.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        network.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        model = cv2.dnn_DetectionModel(network)
        model.setInputParams(size=(416,416), scale=1/255, swapRB=True)
        classes, scores, boxes = model.detect(self.image, self.conf_threshold, self.nms_threshold)
        
        return classes, scores, boxes


    def inference_run(self, image):

        start = time.time()
        classes, scores, boxes = self.inference_dnn(self.path_to_cfg, self.path_to_weights)
        end = time.time()
        
        frame_time = (end-start) * 1000
        FPS = 1.0 * (end-start)

        ratio_h = self.new_h / self.origingal_h
        ratio_w = self.new_w / self.origingal_w

        for (class_id, score, box) in zip(classes, scores, boxes):
            box[0] = int(box[0] * ratio_w) 
            box[1] = int(box[1] * ratio_h)
            box[2] = int(box[2] * ratio_w) 
            box[3] = int(box[3] * ratio_h)

            cv2.rectangle(image, box, (0,255,0), 2)
            label = f'Frame time: {frame_time} ms. FPS:  {FPS:.2f}. ID {self.class_labels[class_id], score}'
            cv2.putText(image, label, (box[0]-30, box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

        cv2.imshow(f'Image Detected: {image}')
        cv2.waitKey(5000)
        cv2.destryAllWindows()

def main():
    path_to_labels = '/workdir/yolov4-inference/coco-classes.txt'
    image_path = '/workdir/yolov4-inference/images'
    path_to_cfg_yolov4 = '/workdir/yolov4-inference/yolov4.cfg'
    path_to_weights_yolov4 = '/workdir/yolov4-inference/yolov4.weights'

    YoloV4DNN(0.3, 0.38, path_to_classes, image_path, path_to_cfg_yolov4, path_to_cfg_yolov4)