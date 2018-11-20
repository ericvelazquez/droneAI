import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
from pathlib import Path
import argparse
import cv2

from VideoGet import VideoGet
from VideoShow import VideoShow
from VideoProcess import VideoProcess

sys.path.append("..")
from utils import label_map_util


### Variables

# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90

# Using drone bool
DRONE_ON = False

### Check if drone's flag is ON
parser = argparse.ArgumentParser()
parser.add_argument('--drone', action='store_true')
options = parser.parse_args()
if options.drone:
    DRONE_ON = True

### Download Model

my_file = Path(MODEL_NAME + "/frozen_inference_graph.pb")
if my_file.is_file() == False:  # Element not downloaded yet
    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    tar_file = tarfile.open(MODEL_FILE)
    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, os.getcwd())

### Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

### Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def multithread(source = 0):

    video_getter = VideoGet(source).start()
    video_processer = VideoProcess(frame=video_getter.frame, detection_graph=detection_graph, category_index = category_index).start()
    #video_shower = VideoShow(video_getter.frame).start()
    while True:
        if video_getter.stopped or video_processer.stopped:
            video_shower.stop()
            #video_processer.stop()
            video_getter.stop()
            break
        video_processer.image_np = video_getter.frame
        cv2.imshow('object detection', cv2.resize(video_processer.image_np, (800, 600)))
        cv2.waitKey(1)
        #video_shower.frame = frame




multithread(0)