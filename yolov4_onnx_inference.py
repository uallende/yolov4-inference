import os
import cv2
import time

class Yolo4DNN:

    def __init__(self, nms_threshold, conf_threshold, class_labels, video_path, path_to_config, path_to_weights):      

        self.nms_threshold = nms_threshold
        self.conf_threshold = conf_threshold
        self.class_labels = class_labels
        self.video_path = video_path
        self.path_to_config = path_to_config
        self.path_to_weights = path_to_weights

        with open(class_labels, 'r') as read_class:
            self.classes = [label.strip() for label in read_class.readlines()]

        # self.frames = self.load_images(self.video_file)

        # for self.frame in self.frames:
        #     self.image = cv2.imread(self.frame)
        #     self.original_h, self.original_w, _ = self.image.shape
        #     dimension = (640, 640)
        #     self.resize_img = cv2.resize(self.image, dimension, interpolation=cv2.INTER_AREA)
        #     self.new_h, self.new_w, _ = self.resize_img.shape
        #     self.inference_run(self.resize_img)

    # def load_images(self, video_file):

    #     img_list = []

    #     for img_original in os.listdir(video_file):
    #         if img_original.endswith('.jpg') or img_original.endswith('.jpeg') or img_original.endswith('.png'):

    #             img_full_path = os.path.join(video_file, img_original)
    #             img_list.append(img_full_path)
            
    #     return img_list

    def inference_dnn(self, frame, path_to_cfg, path_to_weights):

        network = cv2.dnn.readNet(path_to_cfg, path_to_weights)
        network.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        network.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        model = cv2.dnn_DetectionModel(network)
        model.setInputParams(size=(416,416), scale=1/255, swapRB=True)
        classes, scores, boxes = model.detect(frame, self.conf_threshold, self.nms_threshold)
        
        return classes, scores, boxes


    def inference_run(self):

        video_cap = cv2.VideoCapture(self.video_path)

        if (video_cap.isOpened()==False):
            print(f'Video File reading unsuccessful')

        while(video_cap.isOpened()):

            grabbed, frame = video_cap.read()
            frame = cv2.resize(frame, (800, 800))

            if not grabbed:
                exit()

            start = time.time()
            classes, scores, boxes = self.inference_dnn(frame, self.path_to_config, self.path_to_weights)
            end = time.time()
            
            frame_time = (end-start) * 1000
            FPS = 1.0 * (end-start)

            # ratio_h = self.new_h / self.original_h
            # ratio_w = self.new_w / self.original_w

            for (class_id, score, box) in zip(classes, scores, boxes):
                # box[0] = int(box[0] * ratio_w) 
                # box[1] = int(box[1] * ratio_h)
                # box[2] = int(box[2] * ratio_w) 
                # box[3] = int(box[3] * ratio_h)

                cv2.rectangle(frame, box, (0,255,0), 2)
                label = f'Frame time: {frame_time} ms. FPS:  {FPS:.2f}. ID {self.class_labels[class_id], score}'
                cv2.putText(frame, label, (box[0]-30, box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

            cv2.imshow('Image Detected:', frame)
            cv2.waitKey(2000)
            cv2.destryAllWindows()

def main():
    path_to_labels = '/workdir/yolov4-inference/coco-classes.txt'
    video_path = '/workdir/yolov4-inference/videos/Demo_Yolov4.mp4'
    path_to_cfg_yolov4 = '/workdir/yolov4-inference/yolov4.cfg'
    path_to_weights_yolov4 = '/workdir/yolov4-inference/yolov4.weights'

    yolov4_inst = Yolo4DNN(0.3, 0.38, path_to_labels, video_path, path_to_cfg_yolov4, path_to_weights_yolov4)
    yolov4_inst.inference_run()

if __name__ == "__main__":
    main()