from threading import Thread
import sys
import tensorflow as tf
import numpy as np
sys.path.append("..")
from utils import visualization_utils as vis_util

class VideoProcess:
    """
    Class that continuously process a frame using a dedicated thread.
    """

    def __init__(self, detection_graph=None, category_index = None):
        self.detection_graph = detection_graph
        self.category_index = category_index
        self.stopped = False
        self.image_np = None
        self.frame = None
        self.persons = None

    def start(self):
        Thread(target=self.process, args=()).start()
        return self

    def update_frame(self, new_frame):
        self.frame = new_frame

    def person_counter(self, classes,scores, category_index):
        """
        Take output of the NN and counts how many persons are
        in the image
        """
        person_counter = 0
        for index, value in enumerate(classes[0]):
            if scores[0, index] > 0.5: #threshold
                if (category_index.get(value)).get('name').encode('utf8') == 'person':
                    person_counter += 1
        return "Persons: " + str(person_counter)

    def process(self):
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                while not self.stopped:
                    if self.frame is None:
                        self.stop()
                    frame_process = self.frame
                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(frame_process, axis=0)
                    image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                    # Each box represents a part of the image where a particular object was detected.
                    boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                    # Each score represent how level of confidence for each of the objects.
                    # Score is shown on the result image, together with the class label.
                    scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                    classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                    num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
                    # Actual detection.
                    (boxes, scores, classes, num_detections) = sess.run(
                        [boxes, scores, classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})

                    # Visualization of the results of a detection.
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        frame_process,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        self.category_index,
                        use_normalized_coordinates=True,
                        line_thickness=8)

                    self.image_np = frame_process
                    self.persons = self.person_counter(classes, scores, self.category_index)

    def stop(self):
        self.stopped = True