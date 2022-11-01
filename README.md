
# Overview

This repository contains the code for the paper [**Perspective Flow Aggregation for Data-Limited 6D Object Pose Estimation**](https://arxiv.org/abs/2203.09836). Yinlin Hu, Pascal Fua, and Mathieu Salzmann. ECCV 2022.

<p align="center">
  <img src="./images/compare.png">
  <br>
  <em>Different pose refinement paradigms. (a) Given an initial pose P0, existing refinement strategies estimate a pose difference ∆P0 from the input image and the image rendered according to P0, generating a new intermediate pose P1. They then iterate this process until it converges to the final pose Pˆ. This strategy relies on estimating a delta pose from the input images by extracting global object features. These features contain high-level information, and we observed them not to generalize well across domains. (b) By contrast, our strategy queries a set of discrete poses {P1, P2, P3, . . . } that are near the initial pose P0 from pre-rendered exemplars, and computes the final pose Pˆ in one shot by combining all the correspondences {Ci} established between the exemplars and the input. Estimating dense 2D-to-2D local correspondences forces the supervision of our training to occur at the pixel-level, not at the image-level as in (a). This makes our DNN learn to extract features that contain lower-level information and thus generalize across domains. In principle, our method can easily be extended into an iterative strategy, using the refined pose as a new initial one. However, we found a single iteration to already be sufficiently accurate.</em>
</p>

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
* We use online rendering (Pytorch3d) in this repo, which is the version used in the best single-model method in [BOP challenge 2022](https://bop.felk.cvut.cz/home/).
