##############################################################################
# main_face_detect.py - Face detection using either haar cascades method
#                       or YOLOv3 depending on user input. Once faces are
#                       detected a Gaussian blur is applied to their face so
#                       that their privacy is preserved.
#
# Copyright (C) 2019   Humza Syed
#
# This is free software so please use it however you'd like! :)
##############################################################################

import argparse
import os
from haar_detect import haar_face_detection
from yolo_detect import yolo_face_detection


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

    class NiceFormatter(argparse.ArgumentDefaultsHelpFormatter,
                        argparse.RawDescriptionHelpFormatter):
        """Nice-looking formatter for argparse parsers. Prints defaults in help
        messages and allows for newlines in description/epilog."""
        pass

    parser = argparse.ArgumentParser(
        description='Background_Face_Privacy arguments.',
        formatter_class=NiceFormatter)

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
                        help='Specifies if using haar detection or yolo detection where yolo is more accurate, '
                             'Options: 0=Haar, 1=YOLO')

    # arguments to locate XML file and image directory
    parser.add_argument('-u', '--username', type=str, default='test',
                        help='This is the username to help locate your frontal face xml file in opencv for face detection.'
                             'Usage leads to obtaining the path, for example mine is: '
                             '/home/flarelink/opencv/data/haarcascades/haarcascade_frontalface_default.xml'
                             'where flarelink is me the user. Alternatively you can just download the xml in the repository '
                             'but this way will help you locate any others like if you wanted to do eye detection :)')
    parser.add_argument('-x', '--xml_file', type=str, default='haarcascade_frontalface_default.xml',
                        help='Uses xml file specified')
    parser.add_argument('-i', '--img_dir_path', type=str, default='images',
                        help='Uses the provided directory name as the target directory where all input images are')

    # arguments for haar cascade face detection parameters
    parser.add_argument('--scaling', type=float, default=1.1,
                        help='Defines scaling that compensates for larger/smaller faces (similar to a tolerance)')
    parser.add_argument('--size', type=int, default=10,
                        help='Defines minimum sizes of faces')

    # arguments for yolo face detection parameters
    parser.add_argument('-y', '--yolo_path', type=str, default='yolov3-face.cfg',
                        help='Uses the provided path to the yolo config file')
    parser.add_argument('-w', '--weights', type=str, default='yolov3-wider_16000.weights',
                        help='Uses the provided path to the weights file for yolo')
    parser.add_argument('-c', '--classes', type=str, default='yolov3_classes.txt',
                        help='Uses the provided path to the classes text file for yolo')

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
    if (args.username == 'test'):
        xmlPath = os.path.join(os.getcwd(), args.xml_file)
    else:
        xmlPath = os.path.join(os.path.sep, 'home', args.username, 'opencv', 'data', 'haarcascades', args.xml_file)

    # join input image path
    imgsPath = os.path.join(os.getcwd(), args.img_dir_path)

    # check which detection method we're using and creat folder for it if it doesn't exist already
    # then run face detection
    # Haar cascade detection
    if (args.detect == 0):
        if (os.path.exists(os.path.join(os.getcwd(), 'out_images_haar')) == False):
            os.mkdir('out_images_haar')
        print('Haar face detection commencing')
        haar_face_detection(imgsPath, xmlPath, args.scaling, args.size)

    # YOLO detection
    elif (args.detect == 1):
        if (os.path.exists(os.path.join(os.getcwd(), 'out_images_yolo')) == False):
            os.mkdir('out_images_yolo')
        print('YOLO face detection commencing')
        yolo_face_detection(imgsPath, args.yolo_path, args.weights, args.classes)

    else:
        return IOError(
            "Invalid input, please enter a valid input. Check the program's help command for additional details."
            "To check these details run: python3 main_face_detect.py -h")


if __name__ == '__main__':
    main()
