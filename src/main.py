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
from head_pose_estimation import Model_Head
from gaze_estimation import Model_Gaze
from mouse_controller import MouseController


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

    h_model = "D:/Work/IntelNanodegreeIoT/Computer_Pointer_Control/Operations/models/head-pose-estimation-adas-0001/head-pose-estimation-adas-0001"
    head_det = Model_Head(h_model)
    head_det.load_model()

    g_model = "D:/Work/IntelNanodegreeIoT/Computer_Pointer_Control/Operations/models/gaze-estimation-adas-0002/gaze-estimation-adas-0002"
    gaze_det = Model_Gaze(g_model)
    gaze_det.load_model()

    m_control = MouseController('high', 'fast')

    print('Mouse Controller initialized')

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

            fwc = cx.shape[1]
            fhc = cx.shape[0]

            x0 = int(up_l_output[0][0][0][0]*fwc)
            y0 = int(up_l_output[0][1][0][0]*fhc)
            x1 = int(up_l_output[0][2][0][0]*fwc)
            y1 = int(up_l_output[0][3][0][0]*fhc)
            x2 = int(up_l_output[0][4][0][0]*fwc)
            y2 = int(up_l_output[0][5][0][0]*fhc)
            x3 = int(up_l_output[0][6][0][0]*fwc)
            y3 = int(up_l_output[0][7][0][0]*fhc)
            x4 = int(up_l_output[0][8][0][0]*fwc)
            y4 = int(up_l_output[0][9][0][0]*fhc)
            print((x0, y0), (x1, y1), (x2, y2), (x3, y3), (x4, y4))
            # cv.circle(cx, (x0, y0), 5, (46, 164, 79), -1)
            # cv.circle(cx, (x1, y1), 5, (46, 164, 79), -1)

            cv.rectangle(cx, (x0-25, y0-25), (x0+25, y0+25), (0, 255, 0), 3)
            cv.rectangle(cx, (x1-25, y1-25), (x1+25, y1+25), (0, 255, 0), 3)

            left_eye = cx[y0-30:y0+30, x0-30:x0+30]
            right_eye = cx[y1-30:y1+30, x1-30:x1+30]

            '''
            
            
            cv.circle(cx, (x2, y2), 5, (46, 164, 79), -1)
            cv.circle(cx, (x3, y3), 5, (46, 164, 79), -1)
            cv.circle(cx, (x4, y4), 5, (46, 164, 79), -1)

            '''

            '''
            
            Recieved Landmark. Moving on to Head Pose detection
            
            '''

            yaw, pitch, roll = head_det.predict(cx)

            print('Head Pose Angle : ', 'Yaw =', yaw[0][0],
                  'Pitch =', pitch[0][0], 'Roll =', roll[0][0])

            '''
            
            Recieved Angle. Moving on to Gaze detection
            
            '''

            # left_eye_image = gaze_det.preprocess_input(left_eye)
            # print("Left Eye Complete")
            # right_eye_image = gaze_det.preprocess_input(right_eye)
            # print("Right Eye Complete")
            # head_pose_angles = [yaw[0][0], pitch[0][0], roll[0][0]]
            # gaze_vector = gaze_det.predict(
            #     left_eye_image, right_eye_image, head_pose_angles)
            # print("Gaze Vector :", gaze_vector[0])
            # m_control.move(gaze_vector[0], gaze_vector[1])

            try:

                left_eye_image = gaze_det.preprocess_input(left_eye)
                print("Left Eye Complete")
                right_eye_image = gaze_det.preprocess_input(right_eye)
                print("Right Eye Complete")
                head_pose_angles = [yaw[0][0], pitch[0][0], roll[0][0]]
                gaze_vector = gaze_det.predict(
                    left_eye_image, right_eye_image, head_pose_angles)

                print("Gaze Vector :", gaze_vector[0])

                m_control.move(gaze_vector[0][0], gaze_vector[0][1])

            except:
                print("Come closer to the camera... difficuly in prediction")

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
