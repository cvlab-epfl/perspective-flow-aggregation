
# Overview

This repository contains the code for the paper [**Perspective Flow Aggregation for Data-Limited 6D Object Pose Estimation**](https://arxiv.org/abs/2203.09836). Yinlin Hu, Pascal Fua, and Mathieu Salzmann. ECCV 2022.

<p align="center">
  <img src="./images/overview.png">
  <br>
  <em>From Optical Flow to Pose Refinement. After obtaining an exemplar based on the initial pose, we estimate dense 2D-to-2D correspondences between the exemplar and the input image within their respective region of interest. This implicitly generates a set of 3D-to-2D correspondences, which can be used to obtain the final pose by PnP solvers.</em>
</p>

# How to Use

* Download [LINEMOD data](https://u.pcloud.link/publink/show?code=XZeRguVZTFv4x0yBf5fva1ALHiTC9Y1CzWey), [pose initializations](https://u.pcloud.link/publink/show?code=XZfRguVZMzhhbnvKm6QkvcFyIsdwlhrl352y), and [pretrained model](https://u.pcloud.link/publink/show?code=XZDRguVZHYfOMIyuH6ma8cFjFCi2dh0yBRDy).

* Extract LINEMOD into "data" in the current directory.

* Have fun!

# Citing

```
@inproceedings{hu2022pfa,
  title={Perspective Flow Aggregation for Data-Limited 6D Object Pose Estimation},
  author={Yinlin Hu and Pascal Fua and Mathieu Salzmann},
  booktitle={ECCV},
  year={2022}
}
```

# Notes

* The pose initializations come from [WDR-Pose](https://github.com/cvlab-epfl/wide-depth-range-pose).
* We use online rendering here (Pytorch3d)
