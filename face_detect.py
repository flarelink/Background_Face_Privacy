##############################################################################
# face_detect.py - Face detection using either haar cascades method or YOLOv3
#                  depending on user input. Once faces are detected a 
#                  Gaussian blur is applied to their face so that their
#                  privacy is preserved.
#
# Copyright (C) 2019   Humza Syed
#
# This is free software so please use it however you'd like! :)
##############################################################################

import cv2
import numpy as np
import argparse
import os


"""
##############################################################################
# Haar Face detection
##############################################################################
"""
def haar_face_detection(imgsPath, xmlPath, scaling, size):
    """
    Detects faces in images and draws a bounding box around the faces using haar cascading method

    :param imgsPath: Path to the input images directory
    :param xmlPath : Path to the xml file for detecting the front of people's faces
    :param scaling : Parameter for classifier to determine the tolerance of small/big faces in image
    :param size    : Parameter for classifier to know what the minimum sized face is

    :returns:

    (Doesn't return anything except outputted images to the directory 'output_images')
    """

    # keep track of how many images processed on
    counter = 0

    for image_file in (os.listdir(imgsPath)):
        
        # image path
        image_path = os.path.join(imgsPath, image_file)

        # read input image
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # face and eye detection classifiers using haar features
        face_classifier = cv2.CascadeClassifier(xmlPath)

        # detect faces
        faces = face_classifier.detectMultiScale(
                gray_image,
                scaleFactor = scaling,
                minNeighbors = 10,
                minSize = (size, size),
                flags = cv2.CASCADE_SCALE_IMAGE
            )

        # draw rectangles around faces
        for (x, y, w, h) in faces:
            # draw the bounding box - commented out because I don't want a green box 
            # around the faces, but left here if the user wants to add this in
            #cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)

            # apply gaussian blur on faces
            face = image[y:y+h, x:x+w]
            face = cv2.GaussianBlur(face, (23, 23), 30)

            # put blurred face on new image
            image[y:y+face.shape[0], x:x+face.shape[1]] = face

        #cv2.imshow('Faces found', image)

        # strip file extension of original image so we can write similar output image
        # i.e.) people.png --> people_output.png
        image_file = os.path.splitext(image_file)[0]
        cv2.imwrite(os.path.join(os.getcwd(), 'out_images_haar', (image_file + '_output.png')), image)
        #cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        print('Image {} was completed!'.format(counter))
        counter += 1

    print('Processed all {} images! :D'.format(counter))


"""
##############################################################################
# YOLOv3 Face detection
##############################################################################
"""
def yolo_face_detection(imgsPath, yolo_path, weights_file, classes_file):
    """
    Detects faces in images and draws a bounding box around the faces using yolo method

    :param imgsPath:    Path to the input images directory
    :param yolo_path:   Path to the yolo algorithm's config file
    :param weights_file:Pre-trained face weights for yolo algorithm 
    :param classes_file:Classes text file for yolo algorithm 

    :returns:

    (Doesn't return anything except outputted images to the directory 'output_images')
    """
    # keep track of how many images processed on
    counter = 0

    for image_file in (os.listdir(imgsPath)):
        
        # image path
        image_path = os.path.join(imgsPath, image_file)

        # read input image
        image = cv2.imread(image_path)

        # get width and height
        height = image.shape[0]
        width  = image.shape[1]
        scale = 0.00392

        # get class names
        with open(classes_file, 'r') as f:
            classes = [line.strip() for line in f.readlines()]

        # we just have 1 class which is a face
        colors_list = [(0, 255, 0)]

        # read pre-trained model and config file to create network
        net = cv2.dnn.readNet(weights_file, yolo_path)

        # Prepare image to run through network
        # create input blob
        blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)

        # set input blob for the network
        net.setInput(blob)

        # run inference
        outs = net.forward(get_output_layers(net))

        # initializations for detection
        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.35 #0.5
        nms_threshold = 0.4

        # get confidences, bounding box params, class_ids for each detection
        # ignore weak detections (under 0.5 confidence)
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if(confidence > 0.35):#0.5):
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = center_x - w/2
                    y = center_y - h/2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        # apply non-max suppresion
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        # detections after nms
        for ind in indices:
            ind = ind[0]
            box = boxes[ind]
            x = max(0, round(box[0]))
            y = max(0, round(box[1]))
            w = max(0, round(box[2]))
            h = max(0, round(box[3]))

            # draw the bounding box - commented out because I don't want a green box 
            # around the faces, but left here if the user wants to add this in
            #draw_bounding_box(image, classes, class_ids[ind], colors_list, x, y, w, h)

            # draw the blurred bounding box
            blur_detected_object(image, x, y, w, h)

        # strip file extension of original image so we can write similar output image
        # i.e.) people.png --> people_output.png
        image_file = os.path.splitext(image_file)[0]
        cv2.imwrite(os.path.join(os.getcwd(), 'out_images_yolo', (image_file + '_output.png')), image)
        cv2.destroyAllWindows()
    
        print('Image {} was completed!'.format(counter))
        counter += 1

    print('Processed all {} images! :D'.format(counter))


def get_output_layers(net):
    """
    Obtain output layer names in the architecture

    :param   net: yolo network

    :return  output_layers: output layers used in the net
    """

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_bounding_box(image, classes, class_id, colors_list, x, y, w, h):
    """
    Draw bounding boxes in image based off object detected and apply a blur to the inside of the box

    :param  image:      read in image using opencv
    :param  classes:    list of classes
    :param  class_id:   specific class id
    :param  colors:     colors used for bounding box
    :param  x, y, w, h: dimensions of image

    :return (the class label and colored bounding box on the image)

    """

    # get label and color for class
    label = str(classes[class_id])
    color = colors_list[class_id]
    
    # draw bounding box with text over it
    cv2.rectangle(image, (x,y), ((x+w), (y+h)), color, 2)
    cv2.putText(image, label, (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def blur_detected_object(image, x, y, w, h):
    """
    Apply a blur to the detected objects in the image

    :param  image:      read in image using opencv
    :param  x, y, w, h: dimensions of image

    :return (the blurred bounding box on the image)

    """
    # apply gaussian blur on faces
    face = image[y:y+h, x:x+w]
    face = cv2.GaussianBlur(face, (23, 23), 30)

    # put blurred face on new image
    face_y = face.shape[0]
    face_x = face.shape[1]

    image[y:y+face_y, x:x+face_x] = face

"""
##############################################################################
# Parser
##############################################################################
"""
def create_parser():
    """
    Function to take in input arguments from the user

    return: parser inputs
    """
    parser = argparse.ArgumentParser(
        description='Background_Face_Privacy arguments.')

    # function to allow parsing of true/false boolean
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    # determine if user wants to use haar classifier or yolo classifier
    parser.add_argument('-d', '--detect', type=int, default=1,
            help='Specifies if using haar detection or yolo detection where yolo is more accurate, Options: 0=Haar, 1=YOLO; default=1')

    # arguments to locate XML file and image directory
    parser.add_argument('-u', '--username', type=str, default='test',
            help='This is the username to help locate your frontal face xml file in opencv for face detection. Usage leads to obtaining the path, for example mine is: /home/flarelink/opencv/data/haarcascades/haarcascade_frontalface_default.xml where flarelink is me the user. Alternatively you can just download the xml in the repository but this way will help you locate any others like if you wanted to do eye detection :) ; default=test')
    parser.add_argument('-x', '--xml_file', type=str, default='haarcascade_frontalface_default.xml',
            help='Uses xml file specified; default=haarcascade_frontalface_default.xml')
    #parser.add_argument('--img_path', type=str, default='test.jpeg',
            #help='Uses path of image; default=test.jpeg')
    parser.add_argument('-i', '--img_dir_path', type=str, default='images',
            help='Uses the provided directory name as the target directory where all input images are; default=images')
    
    # arguments for haar cascade face detection parameters
    parser.add_argument('--scaling', type=float, default=1.1,
            help='Defines scaling that compensates for larger/smaller faces (similar to a tolerance); default=1.1')
    parser.add_argument('--size', type=int, default=10,
            help='Defines minimum sizes of faces; default=10')

    # arguments for yolo face detection parameters
    parser.add_argument('-y', '--yolo_path', type=str, default='yolov3-face.cfg',
            help='Uses the provided path to the yolo config file; default=yolov3-face.cfg')
    parser.add_argument('-w', '--weights', type=str, default='yolov3-wider_16000.weights',
            help='Uses the provided path to the weights file for yolo; default=yolov3-wider_16000.weights')
    parser.add_argument('-c', '--classes', type=str, default='yolov3_classes.txt',
            help='Uses the provided path to the classes text file for yolo; default=yolov3_classes.txt')

    args = parser.parse_args()

    return args


"""
##############################################################################
# Main, where all the magic starts~
##############################################################################
"""
def main(): 
    """
    Takes in input image to detect faces and then apply a blur like effect of some kind
    """

    # load parsed arguments
    args = create_parser()

    # obtain path to xml based off username
    # if username left as test then load the xml downloaded from repository
    if(args.username == 'test'):
        xmlPath = os.path.join(os.getcwd(), args.xml_file)
    else:
        xmlPath = os.path.join(os.path.sep, 'home', args.username, 'opencv', 'data', 'haarcascades', args.xml_file)

    # join input image path
    imgsPath = os.path.join(os.getcwd(), args.img_dir_path)

    # check which detection method we're using and creat folder for it if it doesn't exist already
    # then run face detection
    # Haar cascade detection
    if(args.detect == 0):
        if(os.path.exists(os.path.join(os.getcwd(), 'out_images_haar')) == False):
            os.mkdir('out_images_haar')
        print('Haar face detection commencing')
        haar_face_detection(imgsPath, xmlPath, args.scaling, args.size)

    # YOLO detection
    elif(args.detect == 1):
        if(os.path.exists(os.path.join(os.getcwd(), 'out_images_yolo')) == False):
            os.mkdir('out_images_yolo')
        print('YOLO face detection commencing')
        yolo_face_detection(imgsPath, args.yolo_path, args.weights, args.classes)

    else:
        return IOError("Invalid input, please enter a valid input. Check the program's help command for additional details. (face_detect.py -h)") 

if __name__== '__main__':
    main()
