# Computer Pointer Controller

The project Aims at controlling the mouse pointer using eye movement.

## Project Set Up and Installation

### Prerequisites

You need to have the following things installed in your system for the project to work

- OpenVINO
- CMake
- Python 3
- Microsoft Visual Studio

### Installing

To know how to install and setup openVINO on different platforms you can take a look at [OpenVINO's repository](https://github.com/opencv/dldt).

- First you need to start with running the Command Prompt with Administrator Privileges.
- Then you have to run the setupvars.bat file in it to set up the openVINO environment.
  The command I used for doing so is `D:\IntelSWTools\openvino_2020.1.033\bin\setupvars.bat`
  You will get an output on the screen something like this.

```
Python 3.6.8
ECHO is off.
PYTHONPATH=D:\IntelSWTools\openvino_2020.1.033\deployment_tools\open_model_zoo\tools\accuracy_checker;D:\IntelSWTools\openvino_2020.1.033\python\python3.6;D:\IntelSWTools\openvino_2020.1.033\python\python3;D:\IntelSWTools\openvino_2020.1.033\deployment_tools\model_optimizer;
[setupvars.bat] OpenVINO environment initialized
```

- You can now run your openvino applications

## Demo

You can now run your openvino application using the command `python D:\Work\IntelNanodegreeIoT\Computer_Pointer_Control\Operations\src\main.py`
Make sure you enter the directory in which you have kept the project files.

### The Various Command Line Parameters

- `--device` this param can be used to run the script using FPGA, MYRIAD, MULTI:CPU,GPU, HETERO:FPGA,CPU by default it is set to CPU
- `--input-type` this param is used to specify the type of input you will give to the script. It can have three types of input `video`,`cam`
- `--input` this param is used to specify the path of the input you will give to the script. For example `D:/Work/IntelOpenVINO/videos/Walk_cut.mp4`. For webcam you need to give `0` as input
- `--prob_threshold` you can change the detection threshold if you want. By default the detection threshold is set to `0.5`

## Documentation

You can refer this [project](https://github.com/Chester-King/Intel-Edge-AI-Scholarship-Project/blob/master/README.md#the-various-command-line-parameters) to understand a bit more about how to use command line arguments by the [demo video](https://drive.google.com/open?id=1cIGan87kJsCDwkodEyu0BUJ7j49WNweL)

## Results

First extracting the face using face detection model and then using it to find the facial landmarks and head pose using two seperate models to determine the gaze vector using gaze detection model.

### Edge Cases

In case of lighting changes or the person going out of frame the code will keep on executing and it will ask the person to come back in frame and come closer for better detection.
In case of multiple people coming into frame the code will only recognize first face. In case the first face leaves then second face will be used.
