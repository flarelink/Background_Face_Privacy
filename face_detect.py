import cv2
import numpy as np
import argparse
import os


"""
##############################################################################
# Face detection
##############################################################################
"""
def face_detection(imgPath, xmlPath, scaling, size):
    # read input image
    img = cv2.imread(imgPath)
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

    print('Found {0} faces!'.format(len(faces)))

    # draw rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

    cv2.imshow('Faces found', img)
    cv2.imwrite('output.png', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



"""
##############################################################################
# Parser
##############################################################################
"""
def create_parser():
    """
    return: parser inputs
    """
    parser = argparse.ArgumentParser(
        description='Background_Face_Privacy arguments.')

    # arguments for content and style image locations
    parser.add_argument('--username', type=str, default='test',
            help='This is the username to help locate your frontal face xml file in opencv for face detection. Usage leads to obtaining the path, for example mine is: /home/flarelink/opencv/data/haarcascades/haarcascade_frontalface_default.xml where flarelink is me the user. Alternatively you can just download the xml in the repository but this way will help you locate any others like if you wanted to do eye detection :) ; default=test')

    parser.add_argument('--xml_file', type=str, default='haarcascade_frontalface_default.xml',
            help='Uses xml file specified; default=haarcascade_frontalface_default.xml')

    parser.add_argument('--img_path', type=str, default='test.jpeg',
            help='Uses path of image; default=test.jpeg')
    
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
        
    print(xmlPath)

    # join image path
    imgPath = os.path.join(os.getcwd(), args.img_path)
    print(imgPath)

    # run face detection
    face_detection(imgPath, xmlPath, args.scaling, args.size)
    

if __name__== '__main__':
    main()
