# People detection on video with darkflow

Simple script for people detection at video using darkflow library.

It takes a .mp4 video file and save it with bounding boxex around people.

To make it work you just need to start it and give a path to input and output files.

Examples:

```
python script.py input_video_path output_video_path
```
Net was trained on [VOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/) data only with 16 epochs.

Accuracy may be improved with longer training.


## Dependencies

Python3, tensorflow 1.0, numpy, opencv 3, [darkflow](https://github.com/thtrieu/darkflow)


### Installing

1. Download this repo.

2. Download weights (.ckpt) files from [GoogleDisk](https://drive.google.com/drive/folders/0B9fBTgfmCIjeM0lxQlBXNlBiOGc) and put them to ckpt folder.


## Running the tests

To test script run

```
python script.py 'path_to_derictory'/test/TUD-Campus.mp4 'path_to_derictory'/test/TUD-Campus_result.mp4
```
Script must create file TUD-Campus_result.mp4 with bounding boxex around people.

* Test video from [MOTChallenge](https://motchallenge.net/vis/TUD-Campus)

## Built With
* [Darkflow](https://github.com/thtrieu/darkflow)
* [Tensorflow](http://www.dropwizard.io/1.0.2/docs/)
* [OpenCV](http://opencv.org/)

