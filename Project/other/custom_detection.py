
##### Author - Nilesh Chopda

##### Project - Traffic Light Detection and Color Recognition using Tensorflow Object Detection API


### Import Important Libraries

import numpy as np
import os
import six.moves.urllib as urllib
import tarfile
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
from os import path
from utils import label_map_util
from utils import visualization_utils as vis_util
import time
import cv2

### Loading Image Into Numpy Array

def load_image_into_numpy_array(image):
    if image.mode not in ('RGB', 'RGBA'):
        image = image.convert('RGB')
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)



### Read Traffic Light objects
# Here,we will write a function to detect TL objects and crop this part of the image to recognize color inside the object. We will create a stop flag,which we will use to take the actions based on recognized color of the traffic light.
def standardize_input(image):
    ## Resize image and pre-process so that all "standard" images are the same size
    if isinstance(image, Image.Image):
        image = np.array(image)
    standard_im = cv2.resize(image.astype('uint8'), dsize=(32, 32))

    return standard_im

def read_traffic_lights_object(image, boxes, scores, classes, max_boxes_to_draw=20, min_score_thresh=0.95,
                               traffic_ligth_label=10,index=0, padding=3):
    im_width, im_height = image.size
    stop_flag = False
    img_i = 0
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores[i] > min_score_thresh and classes[i] == traffic_ligth_label:
            ymin, xmin, ymax, xmax = tuple(boxes[i].tolist())
            (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                          ymin * im_height, ymax * im_height)
            crop_img = image.crop((left-padding, top-padding, right+padding, bottom+padding))

            crop_img = standardize_input(crop_img)
            # plt.imshow(crop_img)

            # save video frames
            crop_img_bgr = cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR)
            os.makedirs('../data/video_processing/temp_crop/frame_{:04d}'.format(index), exist_ok=True)
            cv2.imwrite('../data/video_processing/temp_crop/frame_{:04d}/{:04d}.png'.format(index, img_i), crop_img_bgr)
            img_i += 1
    return



### Function to Plot detected image

def plot_origin_image(image_np, boxes, classes, scores, confidences, prediction_colors):

    # Size of the output images.
    IMAGE_SIZE = (12, 8)
    vis_util.customvisualize_boxes_and_labels(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        confidences,
        prediction_colors,
        min_score_thresh=.95,
        use_normalized_coordinates=True,
        line_thickness=3)
    # plt.figure(figsize=IMAGE_SIZE)
    # plt.imshow(image_np)

    # save augmented images into hard drive
    # plt.savefig( 'output_images/ouput_' + str(idx) +'.png')
    # plt.show()


### Function to Detect Traffic Lights and to Recognize Color

def detect_traffic_lights(PATH_TO_FRAME_IMAGES_DIR, MODEL_NAME, Num_images, padding=3):
    """
    Detect traffic lights and draw bounding boxes around the traffic lights
    :param PATH_TO_FRAME_IMAGES_DIR: testing image directory
    :param MODEL_NAME: name of the model used in the task
    :return: commands: True: go, False: stop
    """

    # --------frame images------
    FRAME_IMAGE_PATHS = [
        os.path.join(PATH_TO_FRAME_IMAGES_DIR, 'frame_{:04d}.png'.format(i)) for i in range(Num_images)
    ]

    # What model to download
    MODEL_FILE = MODEL_NAME + '.tar.gz'
    DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

    # List of the strings that is used to add correct label for each box.
    # PATH_TO_LABELS = 'working_model_clean_data/mscoco_label_map.pbtxt'
    PATH_TO_LABELS = 'other/mscoco_label_map.pbtxt'

    # number of classes for COCO dataset
    NUM_CLASSES = 90

    # --------Download model----------
    if path.isdir(MODEL_NAME) is False:
        opener = urllib.request.URLopener()
        opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
        tar_file = tarfile.open(MODEL_FILE)
        for file in tar_file.getmembers():
            file_name = os.path.basename(file.name)
            if 'frozen_inference_graph.pb' in file_name:
                tar_file.extract(file, os.getcwd())

    # --------Load a (frozen) Tensorflow model into memory
    detection_graph = tf.compat.v1.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    # ---------Loading label map
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map,
                                                                max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)

    with detection_graph.as_default():
        with tf.compat.v1.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            index_counter = 0
            box_list = [[] for _ in range(Num_images)]
            score_list = [[] for _ in range(Num_images)]
            class_list = []

            for image_path in FRAME_IMAGE_PATHS:
                image = Image.open(image_path)

                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                image_np = load_image_into_numpy_array(image)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                
                box_list[index_counter] = boxes
                score_list[index_counter] = scores
                class_list.append(classes)
                read_traffic_lights_object(image, np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes).astype(np.int32), index=index_counter, padding=padding)
                index_counter += 1
                
    return box_list, class_list, score_list


### Function to add the cropped images in the top left corner of the original image
def add_cropped_images(image, index, prediction_color, confidences):
    """
    Add the cropped images in the top left corner of the original image with labels.
    :param image: original image
    :param index: index of the frame
    :param colors: list of colors for each cropped image
    :param probabilities: list of probabilities for each cropped image
    :return: image with cropped images and labels
    """
    try:
        cropped_images_dir = '../data/video_processing/temp_crop/frame_{:04d}'.format(index)
        cropped_images = [cv2.imread(os.path.join(cropped_images_dir, img)) for img in os.listdir(cropped_images_dir) if img.endswith('.png')]
        
        x_offset = 0
        padding = 10  # Padding between cropped images
        for i, cropped_image in enumerate(cropped_images[:2]):
            if cropped_image is not None:
                cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
                cropped_image = cv2.resize(cropped_image, (cropped_image.shape[1] * 3, cropped_image.shape[0] * 3))
                image[0:cropped_image.shape[0], x_offset:x_offset+cropped_image.shape[1]] = cropped_image
                
                # Add label below the cropped image
                label = f"{prediction_color[i]}: {int(confidences[i])}%"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                font_color = (255, 255, 255)  # White color
                thickness = 1
                label_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
                label_x = x_offset
                label_y = cropped_image.shape[0] + label_size[1] + 10
                
                # Draw a filled rectangle as a background for the label
                background_topleft = (label_x, cropped_image.shape[0] + 5)
                background_bottomright = (label_x + label_size[0], label_y + 5)
                cv2.rectangle(image, background_topleft, background_bottomright, (95, 158, 160), cv2.FILLED)  # CadetBlue background
                
                cv2.putText(image, label, (label_x, label_y), font, font_scale, font_color, thickness, lineType=cv2.LINE_AA)
                
                x_offset += cropped_image.shape[1] + padding
    except FileNotFoundError:
        pass
    
    return image


def annotate_image(image_path, output_image_path, boxes, classes, scores, prediction_color, confidences, plot_flag=True):
    image = Image.open(image_path)
    image_np = load_image_into_numpy_array(image)
    
    if plot_flag:
        plot_origin_image(image_np, boxes, classes, scores, confidences, prediction_color)

    # Add the cropped images in the top left corner of the original image
    index = int(image_path.split('/')[-1].split('_')[1].split('.')[0])
    image_np = add_cropped_images(image_np, index, prediction_color, confidences)

    cv2.imwrite(os.path.join(output_image_path), cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))



### Let's detect Traffic lights in test_images directory

if __name__ == "__main__":
    # Specify number of images to detect
    Num_images = 17

    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    print("Current Directory: ", os.getcwd())

    # Specify test directory path
    PATH_TO_FRAME_IMAGES_DIR = '../data/working_model_clean_data/video_processing/temp_frames'
    
    #Print the amount of files in directory
    print("TEST: ",len([name for name in os.listdir(PATH_TO_FRAME_IMAGES_DIR) if os.path.isfile(os.path.join(PATH_TO_FRAME_IMAGES_DIR, name))]))

    # Specify downloaded model name
    # MODEL_NAME ='ssd_mobilenet_v1_coco_11_06_2017'    # for faster detection but low accuracy
    MODEL_NAME = 'faster_rcnn_resnet101_coco_11_06_2017'  # for improved accuracy

    commands = detect_traffic_lights(PATH_TO_FRAME_IMAGES_DIR, MODEL_NAME, Num_images)
    print(commands)  # commands to print action type, for 'Go' this will return True and for 'Stop' this will return False





