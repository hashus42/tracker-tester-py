# Tracker Tester CPP

Python program for testing and comparing the performance of different object tracking algorithms.

## Test Result

A sample test result video is included below:

[test-in-6-tracker.mp4](test-in-6-tracker.mp4)

## Description

This program uses OpenCV to read a video file and apply different tracking algorithms to detect and track objects. The program displays the tracking results in real-time, allowing for easy comparison of the different algorithms.

## Algorithms Used

* Nano
* MIL
* KCF
* MedianFlow
* GOTURN
* CSRT

## Run


### You need to download following files

* * For GOTURN:
* goturn.prototxt and goturn.caffemodel: https://github.com/opencv/opencv_extra/tree/c4219d5eb3105ed8e634278fad312a1a8d2c182d/testdata/tracking
* * For NanoTrack:
* nanotrack_backbone: https://github.com/HonglinChu/SiamTrackers/blob/master/NanoTrack/models/nanotrackv2/nanotrack_backbone_sim.onnx
* nanotrack_headneck: https://github.com/HonglinChu/SiamTrackers/blob/master/NanoTrack/models/nanotrackv2/nanotrack_head_sim.onnx

To download files via terminal you can use following method
```
wget <url>
```


