# People detection on video with darkflow

Simple script for people detection at video using darkflow library.

It takes a .mp4 video file and save it with bounding boxex around people.

To make it work you just need to start it and give a path to input and output files.

Examples:

```
python detector.py -i input_video_path -o output_video_path
```

## Dependencies

Python, tensorflow 1.0, numpy, opencv 3, [darkflow](https://github.com/thtrieu/darkflow)


## Installing

1. Download this repo.

2. Download weights YOLOv2 608x608 weights file from [YOLO](https://pjreddie.com/darknet/yolo/) and put them to bin folder.


## Running the tests

To test script run

```
python detector.py -i 'path_to_input_video_file' -o 'path_to_output_video_file'
```
Script must create file with bounding boxex around people.

Test video was downloaded from [VideoBlocks](https://www.videoblocks.com/video/people-in-hotel-area-with-swimming-pool-in-sharm-el-sheikh-egypt-4ohxnqwtxijwaf9o9/)

Sample of processed video you may download [here](https://drive.google.com/open?id=0B9fBTgfmCIjeOXlkdWFGa0xQbW8)

## Parameters

```
-i - path to input video file.

-o - path to output video file.

-t - confidence level of neural network

-v - Show video in progress

-c - Read input from WebCam
```

## Built With
* [Darkflow](https://github.com/thtrieu/darkflow)
* [Tensorflow](http://www.dropwizard.io/1.0.2/docs/)
* [OpenCV](http://opencv.org/)

# human_detector_sp
