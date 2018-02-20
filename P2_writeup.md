# **Traffic Sign Classifier** 


**Traffic Sign Classifier**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Pipeline of lane lines detection from a video

My pipeline for detecting lane line consists of the following steps. Note that the steps for lane lines detection from an image just skips step 1.

1. Extract sequence of images from a video using moviepy, and apply a process_image function to modify each frame of image.

2. Take the color iamge and convert it to grayscale.

3. Use a Gaussian kernel to smooth out the noisey parts on the image, and apply canny edge detection on the whole image. Note that the Gaussian kernel size and edge thresholds are tuned such that the lane lines can be clearly identified, while other irrelevant details are removed as much as possible.

4. Crop the image to an interested polygon region and mask not intereted region as black. This region is the front part of the car. It extends to the place where the lane lines almost intersect with each other. 

5. Apply hough line detection algorithm to detect all lines in the interested region. The max/min line length/gap parameters are tuned such that most lane lines can be detected with minimal amount of undesired lines.

6. Postprocessing on the detected lines to extract two lane lines. K-mean clustering algorithm is applied by assuming there are two cluster centers. The lines representation are converted back to the hough space representation (rho, theta), which avoids the invalid infinite slope case. Note that K-mean clustering is similar to averaging the (rho, theta) pairs to two cluster centers.

### 2. Potential shortcomings with my current pipeline

After performing this lane line detection project, I realized that self-driving is not as easy as we think. Even the most basic lane line detection problem can not be solved perfectly. The potential shortcomings are:

1. If the car is not parallel to the lane (lane changing and other cases), then the region croping method is not valid at all. If the region is not cropped, a lot of external noise will be introduced. A more robust line extraction algorithm is needed (suggested in the next section).

2. Canny edge detection has some threshold values that is tuned in a handcrafted way. If the lighting condition and other factors changed, it will not behave as expected.

3. Lane lines are not necessarily straigt lines. Ways to detect curved lines are needed.


### 3. Suggest possible improvements to my pipeline

Some possile improvements to my current pipeline are:

1. Use better model of lane lines to detect lane lines, e.g., vanishing point detection using Gaussian probability model in the literature.

2. Hough line detection algorithm can be extended to detect curved lines using the method in the literature.

3. Lane lines might wore out. Since the video is a continuous sequence of images, we can leverage the detected line information in the previous frame to guide detection in the next frame. Higher detection precision can be achieved using the continuity of lanes frame by frame.
