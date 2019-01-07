##############################################################################
# face_detect.py - basic face detection which will further be developed to 
#                  apply an effect over people's faces so that their privacy
#                  is preserved
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
# Face detection
##############################################################################
"""
def face_detection(imgsPath, xmlPath, scaling, size):
    """
    Detects faces in images and draws a bounding box around the faces

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
        img = cv2.imread(image_path)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # face and eye detection classifiers using haar features
        face_classifier = cv2.CascadeClassifier(xmlPath)

        # detect faces
        faces = face_classifier.detectMultiScale(
                gray_img,
                scaleFactor = scaling,
                minNeighbors = 10,
                minSize = (size, size),
                flags = cv2.CASCADE_SCALE_IMAGE
            )

        #print('Found {0} faces!'.format(len(faces)))

        # draw rectangles around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        cv2.imshow('Faces found', img)

        # strip file extension of original image so we can write similar output image
        # i.e.) people.png --> people_output.png
        image_file = os.path.splitext(image_file)[0]
        cv2.imwrite(os.path.join(os.getcwd(), 'out_images', (image_file + '_output.png')), img)
        #cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        counter += 1

    print('Processed all {} images! :D'.format(counter))



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

    # arguments for content and style image locations
    parser.add_argument('--username', type=str, default='test',
            help='This is the username to help locate your frontal face xml file in opencv for face detection. Usage leads to obtaining the path, for example mine is: /home/flarelink/opencv/data/haarcascades/haarcascade_frontalface_default.xml where flarelink is me the user. Alternatively you can just download the xml in the repository but this way will help you locate any others like if you wanted to do eye detection :) ; default=test')

    parser.add_argument('--xml_file', type=str, default='haarcascade_frontalface_default.xml',
            help='Uses xml file specified; default=haarcascade_frontalface_default.xml')

    #parser.add_argument('--img_path', type=str, default='test.jpeg',
            #help='Uses path of image; default=test.jpeg')

    parser.add_argument('--img_dir_path', type=str, default='images',
            help='Uses the provided directory name as the target directory where all input images are; default=images')
    
    parser.add_argument('--scaling', type=float, default=1.1,
            help='Defines scaling that compensates for larger/smaller faces (similar to a tolerance); default=1.1')
    
    parser.add_argument('--size', type=int, default=10,
            help='Defines minimum sizes of faces; default=10')

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
        
    #print(xmlPath)

    # join input image path
    imgsPath = os.path.join(os.getcwd(), args.img_dir_path)
    #print(imgsPath)

    # create output images directory if it doesn't exist
    if(os.path.exists(os.path.join(os.getcwd(), 'out_images')) == False):
        os.mkdir('out_images')

    # run face detection
    face_detection(imgsPath, xmlPath, args.scaling, args.size)
    

if __name__== '__main__':
    main()
