import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
from pathlib import Path
import argparse
import cv2
import ps_drone

from VideoGet import VideoGet
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

### Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

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

def set_up_drone():
  """
  Set ups the AR drone 2.0 and return drone object
  """
  drone = ps_drone.Drone()                                    # Start using drone
  drone.startup()                                             # Connects to drone and starts subprocesses
  drone.reset()                                               # Sets drone's status to good
  while (drone.getBattery()[0] == -1):      time.sleep(0.1)   # Waits until drone has done its reset
  print "Battery: "+str(drone.getBattery()[0])+"%  "+str(drone.getBattery()[1])
  drone.useDemoMode(True)                                     # Just give me 15 basic dataset per second
  drone.setConfigAllID()                                      # Go to multiconfiguration-mode
  drone.setConfig("control:altitude max","1800")
  drone.sdVideo()                                             # Choose lower resolution
  drone.frontCam()                                            # Choose front view
  return drone

class MultiThread:
    """
    This class jpins both VideoGet and VideoProcess classes and also initialize
    and controls the AR Parrot
    """

    def __init__(self, source = 0, ip=-1, parrot=None):
        self.video_getter = VideoGet(source, ip).start()
        self.video_processer = VideoProcess(detection_graph=detection_graph, category_index=category_index).start()
        self.parrot = parrot

    def start(self):
        should_stop = False
        while not should_stop:
            if self.video_getter.stopped or self.video_processer.stopped:
                # self.video_shower.stop()
                self.video_processer.stop()
                self.video_getter.stop()
                should_stop = True

            self.video_processer.update_frame(self.video_getter.frame)

            c = cv2.waitKey(1)
            if self.video_processer.image_np is not None:
                str_person_counter = self.video_processer.persons
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(self.video_processer.image_np, str_person_counter, (10, 30), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow('object detection', cv2.resize(self.video_processer.image_np, (800, 600)))

                if c & 0xFF == ord('0'):
                        self.video_processer.stop()
                        self.video_getter.stop()
                        if self.parrot:
                            self.parrot.stop()
                            self.parrot.land()
                        should_stop = True

            if DRONE_ON and self.parrot:
                if c & 0xFF == ord('t'):
                    self.parrot.takeoff()
                elif c & 0xFF == ord('l'):
                    self.parrot.land()
                elif c & 0xFF == ord('h'):
                    self.parrot.hover()
                elif c & 0xFF == ord('w'):
                    self.parrot.moveForward()
                elif c & 0xFF == ord('s'):
                    self.parrot.moveBackward()
                elif c & 0xFF == ord('a'):
                    self.parrot.moveLeft()
                elif c & 0xFF == ord('d'):
                    self.parrot.moveRight()
                elif c & 0xFF == ord('q'):
                    self.parrot.turnLeft()
                elif c & 0xFF == ord('e'):
                    self.parrot.turnRight()
                elif c & 0xFF == ord('z'):
                    self.parrot.stop()



def main():
    ip = -1
    parrot = None
    if DRONE_ON:
        parrot = set_up_drone()
        ip = "tcp://" + parrot.DroneIP + ":5555" # Connect to drone camera

    MultiThread(0,ip,parrot).start()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()