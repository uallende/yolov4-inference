
file_path = f'/workdir/yolov4-inference/coco-classes.txt'

with open(file_path, 'r') as f:
    classes = [label.strip() for label in f.readlines()]

print(classes)

