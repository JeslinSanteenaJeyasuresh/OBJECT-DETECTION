# OBJECT-DETECTION
**what is OBJECT-DETECTION?**
OBJECT-DETECTION is a type of image recognition which is used to identify and locate the presence of an object.It provides the bounding-box information and the classification of the object.

**what is OBJECT-CLASSIFICATION?**
OBJECT-DETECTION is a type of image recognition which is used for classification of a object like(dog,cat..)

**what is OBJECT-SEGMENTATION?**
OBJECT-SEGMENTATION helps us to segment the distinct objects in a image in the pixel level

Performance metrics in OBJECT-DETECTION:

1.**Intersection over Union (IoU)**:IoU helps evaluate how well the predicted bounding box surrounds the object.
It is a key metric used in **object localization** to measure how accurately a model predicts the location of an object with a bounding box compared to the ground truth (actual bounding box).
It is calculated as the ratio of the area of overlap (intersection) between the predicted and ground truth boxes to the total combined area (union) of both boxes

The IoU score ranges from 0 to 1, where 1 means a perfect match (the predicted box exactly matches the ground truth), and 0 means no overlap at all.

A higher IoU indicates better localization accuracy by the model.

**2.Mean Average Precision (mAP)**:mAP measures how well a model detects objects and classifies them correctly in images, combining ideas of precision and recall over different confidence thresholds.
