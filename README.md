# What is this code for?
The code in this repository facilitates the training of the YOLO v2 (aka YOLO 9000) deep neural network on custom datasets. Currently it only works with the [DeepFashion dataset](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html), but the code in the `training` directory _should_ be pretty general. I tried to make the code work for multiple objects in an image, but DeepFashion only has one bbox per image so it might not work quite right. In the end I still couldn't get it working quite right, so it should be a decent starting point but don't assume it's correct. 

For more information on the model see:

- [Joseph Redmon's blog](https://pjreddie.com/darknet/yolo/)
- The [paper](https://arxiv.org/pdf/1612.08242.pdf) by Joseph Redmon and Ali Farhadi detailing the model. 
- The [paper](https://arxiv.org/pdf/1506.02640.pdf) detailing v1. Summing over anchor boxes and replacing i with ij in the last term of equation 3 recovers the loss function implemented here. 

# Using the code
## Getting the model
- From [Joseph Redmon's blog](https://pjreddie.com/darknet/yolo/), download the cfg and weights files for YOLOv2
- Clone [this tool](https://github.com/allanzelener/YAD2K.git) for converting from Darknet model file format to Keras. 
- Run `./yad2k.py -p` in the cloned repository using the downloaded cfg and weights files as targets. This will output the keras loadable model file, anchor box definitions and a plot of the graph. The plot should match `model_data/yolo.png`.
- Divide each of the anchor dimensions by the grid width or height as appropriate. It should be 13 for width and height if the correct cfg and weights were downloaded. 
- (Optional) Use k-means or Gaussian mixture modeling to find anchor box dimensions better suited to the chosen dataset. 

## Training on new data
Begin by preprocessing all images to be the same height and width, update their bounding boxes to match this definition. The `preprocessing` folder has tools to do this for the DeepFashion dataset.

After preprocessing, you should have a directory containing all your images and a csv file storing the bounding box definitions. 

The bounding box file should have the following fields:

- image\_name: Image file name treating parent directory as root. (ie. `all_images/dogs/terriers/1.jpg` stored as `/dogs/terriers/1.jpg`)
- category\_label: 1-indexed label for the object class
- center\_x, center\_y: Bounding box centers. x, y values specified relative to an origin at the top left of an image. Moving down increases y, moving right increases x. Represented as fraction of image dimensions.
- width\_x, width\_y: Bounding box dimensions. Represented as fraction of image dimensions. 

## TODO's and Issues
- Localization loss only calculated for grid cells containing the image center. May have decent IoU from neighboring grid cells not picked up on. 
- Can only handle one object per grid cell. 
- The Labels class is too problem specific, needs refactoring.
- Code documentation needs updating and improvement.
- The YOLO9000 paper seems to indicate that width and height values are not bounded. This caused problems for me when training. To fix these problems while still using anchor box priors I use the below method for transforming model output:
```
w_s = \sigma(t_w)
b_w = p_w + E[(x - p_w)| x \in [max(0, w_s - .1), min(1, w_s + .1)]
```
Notation follows the YOLO9000 paper. Expectation wrt a normal distribution around p\_w with variance .16. The variance and .1 value above were mostly arbitrary.
They might be worth optimizing. Height is analagous. 
This definition skews width values towards p\_w while guaranteeing they stay between zero and one.
I ended up approximating part of the expectation. b\_w is still roughly \in [0, 1]
but I need to clip values to be sure.
This might be too computationally expensive, maybe there's a simpler way. 


Thanks to [datlife](https://github.com/datlife/yolov2) for a starting point on implementing the loss function. 
