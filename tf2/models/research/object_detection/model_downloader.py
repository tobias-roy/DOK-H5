import wget
# The below is for the resnet
# model_link = "http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz"
# Below is for EffiecientDet D4
# model_link = "http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d4_coco17_tpu-32.tar.gz"
model_link = "http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz"

model_link = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8.tar.gz"
wget.download(model_link)
import tarfile
# tar = tarfile.open('faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz')
# tar = tarfile.open('efficientdet_d4_coco17_tpu-32.tar.gz')
tar = tarfile.open('efficientdet_d0_coco17_tpu-32.tar.gz')
tar = tarfile.open('ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8.tar.gz')
tar.extractall('.')
tar.close()