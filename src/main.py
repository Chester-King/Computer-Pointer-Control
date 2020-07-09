import os
import sys
import time
import socket
import json
import cv2 as cv
from argparse import ArgumentParser
import numpy as np
from input_feeder import InputFeeder
from face_detection import Model_Face
from facial_landmarks_detection import Model_Land


def vid_inp(args):

    print(args.input_type)
    if(args.input_type == 'cam'):
        inp_feed = InputFeeder(args.input_type)
    else:
        inp_feed = InputFeeder(args.input_type, args.input)

    inp_feed.load_data()

    f_model = "D:/Work/IntelNanodegreeIoT/Computer_Pointer_Control/Operations/models/face-detection-adas-binary-0001/face-detection-adas-binary-0001"
    face_det = Model_Face(f_model)
    face_det.load_model()

    l_model = "D:/Work/IntelNanodegreeIoT/Computer_Pointer_Control/Operations/models/landmarks-regression-retail-0009/landmarks-regression-retail-0009"
    land_det = Model_Land(l_model)
    land_det.load_model()

    for x in inp_feed.next_batch():

        up_output = face_det.predict(x)
        p_output, w, h = face_det.preprocess_output(up_output)

        if(p_output == [[]]):
            cv.putText(x, "No Face Present", (0, 20),
                       cv.FONT_HERSHEY_COMPLEX, 0.6, (0, 125, 255), 1)
        else:
            cv.putText(x, "At least one Face Present", (0, 20),
                       cv.FONT_HERSHEY_COMPLEX, 0.6, (0, 125, 255), 1)
        if(p_output == [[]]):
            print('No Face')
        else:
            # print(p_output[0][0], w, h)
            print(x.shape)
            fw = x.shape[1]
            fh = x.shape[0]
            xmin = int(p_output[0][0][3] * fw)
            ymin = int(p_output[0][0][4] * fh)
            xmax = int(p_output[0][0][5] * fw)
            ymax = int(p_output[0][0][6] * fh)
            cx = x[ymin:(ymax + 1), xmin:(xmax + 1)]

            '''
            
            Face Cropped. Moving on to Landmark detection
            
            '''

            up_l_output = land_det.predict(cx)
            x0 = up_l_output[0][0][0][0]
            y0 = up_l_output[0][1][0][0]
            x1 = up_l_output[0][2][0][0]
            y1 = up_l_output[0][3][0][0]
            x2 = up_l_output[0][4][0][0]
            y2 = up_l_output[0][5][0][0]
            x3 = up_l_output[0][6][0][0]
            y3 = up_l_output[0][7][0][0]
            x4 = up_l_output[0][8][0][0]
            y4 = up_l_output[0][9][0][0]
            print((x0, y0), (x1, y1), (x2, y2), (x3, y3), (x4, y4))

        cv.imshow('Window', cx)
        cv.waitKey(30)

    inp_feed.close()


def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-it", "--input-type", required=False,
                        default='cam', type=str, help="Input Type")
    parser.add_argument("-i", "--input", required=False,
                        default="D:/Work/IntelNanodegreeIoT/Computer_Pointer_Control/Operations/bin/demo.mp4", type=str, help="Path to image or video file")

    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def main():
    """
    Load the network and parse the output.
    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()

    # Grab the input
    vid_inp(args)


if __name__ == '__main__':
    main()
