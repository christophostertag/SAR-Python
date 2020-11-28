## SAR-Python: Search and Rescue with Airborne Optical Sectioning implemented in Python with OpenCv and PyTorch

#### German News Article: https://ooe.orf.at/stories/3077237/

#### Paper from our professors and post-docs at JKU: https://arxiv.org/abs/2009.08835

**Abstract**:
  We show that automated person detection under occlusion conditions can be significantly improved by combining multi-perspective images before classification. Here, we employed image integration by Airborne Optical Sectioning (AOS)---a synthetic aperture imaging technique that uses camera drones to capture unstructured thermal light fields---to achieve this with a precision/recall of 96/93%. Finding lost or injured people in dense forests is not generally feasible with thermal recordings, but becomes practical with use of AOS integral images. Our findings lay the foundation for effective future search and rescue technologies that can be applied in combination with autonomous or manned aircraft. They can also be beneficial for other fields that currently suffer from inaccurate classification of partially occluded people, animals, or objects.

#### Youtube Video
[![YOUTUBE VIDEO](https://img.youtube.com/vi/kyKVQYG-j7U/1.jpg)](https://www.youtube.com/watch?v=kyKVQYG-j7U)

### Usage (PROJECT NOT YET COMPLETED):

Download data.zip from ... and unzip.

1. Install requirments
```
pip3 install -r requirments.txt
```

#### Calibration (Optional)
Download calibration.zip from ... and unzip.

To calibrate
```
python3 calibrate.py -r
```
