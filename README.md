# Product Recognition on Store Shelves #
Project work for the UNIBO Computer Vision M course.
Group members:
 - [Gioele Pisanelli](https://github.com/gpisanelli)
 - [Alberto Jesu](https://github.com/albjerto)
 - [Mattia Campestri](https://github.com/mcampestri)
 
 For a detailed report, consult [this document](https://github.com/gpisanelli/ProgettoCV/blob/master/Report_CV.pdf).
 
 ## Introduction ##
 The aim of this project is to develop a Computer Vision algorithm for the recognition of cereal boxes on store shelves, given
  - a set of __Scene__ images, which can be found in this repository at `.\images\scenes`, depicting store shelves with cereal boxes in different setups;
   - a set of __Model__ images, which can be found in this repository at `.\images\models`, representing various cereal boxes, and will be the templates that the algorithm will search for in the scenes.

## Project Structure ##
The scene images are categorized as either *easy*, *medium* or *hard*, depending on the quality of the image, as well as the number of objects represented and the presence of nuisances. Thus, three separated pipelines were developed to take care of the problem at hand.

### Easy pipeline ###
The first subset of scenes contains only a limited number of boxes, each present only one time, without repeated boxes and at a high enough resolution. For this scenario, the pipeline is:

 1. SIFT feature detection and Flann matching
 2. Match validation

The evaluation process is very efficient and does not have a significant impact on execution time.

### Medium pipeline ###
The second subset of images contains a larger number of boxes, with the possibility of multiple instances for each box. The pipeline consists in:

 1. SIFT feature detection and Flann matching
 2. Generalized Hough Transform
 3. Match validation

The adopted strategy yields good results, correctly finding all the cereal boxes in each scene. 

### Hard pipeline ###
The last subset of images represent a very large amount of boxes, around 40, on multiple shelves, with the presence of distractor elements such as the prices tags and are low-resolution. This last pipeline consists in:
 1. Shelf splitting
 2. Sub-scene processing
 3. SIFT feature detection and Flann matching
 4. Generalized Hough Transform
 5. Match validation

Even with some imperfections, the number of boxes correctly labeled is overall satisfying.

## Execution ##
The pipelines can be excuted on the corresponding subsets by using the options
 - `-e` for the *easy* scenes,
 - `-m` for the *medium* scenes,
 - `-h` for the *hard* scenes.
