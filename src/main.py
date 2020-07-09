import os
import sys
import time
import socket
import json
import cv2 as cv
from argparse import ArgumentParser
import numpy as np
from input_feeder import InputFeeder


def vid_inp(args):

    print(args.input_type)
    if(args.input_type == 'cam'):
        inp_feed = InputFeeder(args.input_type)
    else:
        inp_feed = InputFeeder(args.input_type, args.input)

    inp_feed.load_data()

    for x in inp_feed.next_batch():
        print(x)
        cv.imshow('Window', x)
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
