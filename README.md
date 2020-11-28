## SAR-Python: Search and Rescue with Airborne Optical Sectioning implemented in Python with OpenCv and PyTorch

Part of our student project at JKU (@JKU-ICG): https://github.com/JKU-ICG/cvlab-2020

#### German News Article: https://ooe.orf.at/stories/3077237/

#### Paper from our professors and post-docs at JKU: https://arxiv.org/abs/2009.08835 (Published in [Nature](https://www.nature.com/articles/s42256-020-00261-3))

**Abstract**:
  We show that ***automated person detection under occlusion*** conditions can be significantly improved by *combining multi-perspective images* before classification. Here, we employed image integration by ***Airborne Optical Sectioning (AOS)***---a synthetic aperture imaging technique that uses camera drones to capture unstructured thermal light fields---to achieve this with a ***precision/recall of 96/93%***. Finding lost or injured people in dense forests is not generally feasible with thermal recordings, but becomes practical with use of AOS integral images. Our findings lay the foundation for effective future search and rescue technologies that can be applied in combination with ***autonomous or manned aircraft***. They can also be beneficial for other fields that currently suffer from inaccurate classification of partially occluded people, animals, or objects.

#### Youtube Video
[![YOUTUBE VIDEO](https://img.youtube.com/vi/kyKVQYG-j7U/0.jpg)](https://www.youtube.com/watch?v=kyKVQYG-j7U)

#### Illustration
<img src="https://media.springernature.com/m685/springer-static/image/art%3A10.1038%2Fs42256-020-00261-3/MediaObjects/42256_2020_261_Fig1_HTML.png" width="480" alt>

*(a) Our drone autonomously scanning a forest patch. In contrast to recording and analyzing single images (b), AOS combines multiple images that are captured within a synthetic aperture before the resulting integral image is analyzed (c). Single thermal drone recordings at an altitude of 35 m above dense forest ground: (d) mixed forest, (e) conifer forest. The arrows indicate partial heat signals of occluded people on the ground. The insets show AOS results that are achieved when multiple thermal images are integrated. Note that contrast and brightness of the insets have been adjusted for better visibility. [Source](https://arxiv.org/abs/2009.08835)*

### Usage (PROJECT NOT YET COMPLETED):
#### Set-Up

> GET DATA: Download data.zip from [JKU Drive](https://drive.jku.at/filr/public-link/file-download/ff8080827595a3570175b7cd458f44a8/22433/-3426038204355214966/data_SAR.zip) and unzip.

1. Install requirments
```
pip3 install -r requirments.txt
```

#### Calibration (Optional)
All the calibration parameters for the camera are already provided in this Github repository and will be automatically applied during the image preprocessing.
> GET DATA: Download calibration.zip from [Google Drive](https://drive.google.com/open?id=1sn5okDv9zIt2ieGDdhi8-QqPwrsDI4-P) and unzip.

Run
```
python3 calibrate.py -r
```
