import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import cv2
from pathlib import Path
import time
import ps_drone
import argparse


sys.path.append("..")
from utils import label_map_util
from utils import visualization_utils as vis_util



 

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

#Using drone bool
DRONE_ON = False

### Check if drone's flag is ON
parser = argparse.ArgumentParser()
parser.add_argument('--drone', action='store_true')
options = parser.parse_args()
if options.drone:
  DRONE_ON = True
  

### Download Model

my_file = Path(MODEL_NAME+"/frozen_inference_graph.pb")
if my_file.is_file() == False: #Element not downloaded yet
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
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


### Helper code

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)



def person_counter(classes):
  """
  Take output of the NN and counts how many persons are
  in the image
  """
  person_counter = 0
  for index, value in enumerate(classes[0]):
      object_dict = {}
      if scores[0, index] > 0.5:
        if (category_index.get(value)).get('name').encode('utf8') == 'person':
          person_counter += 1
  return "Persons: " + str(person_counter)

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
  
#If we are using the drone, set it up and use its camera
if DRONE_ON: 
  ARdrone = set_up_drone()
  cap = cv2.VideoCapture("tcp://"+ARdrone.DroneIP+":5555")    # Connect to drone camera
else:
  # Capture video from webcam
  cap = cv2.VideoCapture(0) 
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    while True:
      ret, image_np = cap.read()
      if not cap:
        time.sleep(0.1)   
        continue
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      # Actual detection.
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      
      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)

      # Adding person counter to the image 
      str_person_counter = person_counter(classes)
      font =cv2.FONT_HERSHEY_SIMPLEX
      cv2.putText(image_np, str_person_counter, (10, 30), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
      # Show image
      cv2.imshow('object detection', cv2.resize(image_np, (800,600)))

      # Control Drone
      if DRONE_ON and ARdrone:
        key = ARdrone.getKey()
        if key == " ":
          if ARdrone.NavData["demo"][0][2] and not ARdrone.NavData["demo"][0][3]: 
            ARdrone.takeoff()
          else:                               
            ARdrone.land()
        elif key == "0":  
          ARdrone.hover()
        elif key == "w":  
          ARdrone.moveForward()
        elif key == "s":  
          ARdrone.moveBackward()
        elif key == "a":  
          ARdrone.moveLeft()
        elif key == "d":  
          ARdrone.moveRight()
        elif key == "q":  
          ARdrone.turnLeft()
        elif key == "e":  
          ARdrone.turnRight()
        elif key != "":
          ARdrone.stop()
          ARdrone.land()
          cv2.destroyAllWindows()   
          break
          
      if cv2.waitKey(25) & 0xFF == ord('v'):
          cv2.destroyAllWindows()
          break