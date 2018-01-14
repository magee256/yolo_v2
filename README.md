# What is this code for?
The code in this repository facilitates the training of the YOLO v2 (aka YOLO 9000) deep neural network on custom datasets. Currently it only works with the DeepFashion dataset, but the code in the `training` directory _should_ be pretty general. 

For more information on the model see:

- [Joseph Redmon's blog](https://pjreddie.com/darknet/yolo/)
- The [paper](https://arxiv.org/pdf/1612.08242.pdf) by Joseph Redmon and Ali Farhadi detailing the model. 
- The [paper](https://arxiv.org/pdf/1506.02640.pdf) detailing v1. Summing over anchor boxes and replacing i with ij in the last term of equation 3 recovers the loss function implemented here. 

# Using the code
## Getting the model
- From [Joseph Redmon's blog](https://pjreddie.com/darknet/yolo/), download the cfg and weights files for YOLO v2
- Clone [this tool](https://github.com/allanzelener/YAD2K.git) for converting from Darknet model file format to Keras. 
- Run `./yad2k.py -p` in the cloned repository using the downloaded cfg and weights files as targets. This will output the keras loadable model file, anchor box definitions and a plot of the graph. The plot should match `model_data/yolo.png`.
- (Optional) Use k-means or Gaussian mixture modeling to find anchor box dimensions better suited to the chosen dataset. 

## Training on new data
Begin by preprocessing all images to be the same height and width, update their bounding boxes to match this definition. The `preprocessing` folder has tools to do this for the DeepFashion dataset.

After preprocessing, you should have a directory containing all your images and a csv file storing the bounding box definitions. 

The bounding box file should have the following fields:

- image\_name: Image file name treating parent directory as root. (ie. `all_images/dogs/terriers/1.jpg` stored as `/dogs/terriers/1.jpg`)
- category_label: 1-indexed label for the object class
- center\_x, center\_y: Bounding box centers. x, y values specified relative to an origin at the top left of an image. Moving down increases y, moving right increases x. Represented as fraction of image dimensions.
- width\_x, width\_y: Bounding box dimensions. Represented as fraction of image dimensions. 

## TODO's and Issues
- Localization loss only calculated for grid cells containing the image center. May have decent IoU from neighboring grid cells not picked up on. 
- The Labels class is too problem specific, needs refactoring.
- Code documentation needs updating and improvement.


Thanks to [datlife](https://github.com/datlife/yolov2) for a starting point on implementing the loss function. 