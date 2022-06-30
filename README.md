# Inspection of concrete structures

## Author: 

  Bouazzaoui Mohammed
  
  Junior Data Scientist @ Becode
  
  https://www.linkedin.com/in/mbouazzaoui/

## Timeline:

  Start of project : 20/6/2022
  End of project : 30/06/2022

## Used technology

	Deep Neural Network / convolutional neural networks  / Image processing


## Link to app deployed on Heroku

https://concreteinspection.herokuapp.com/info/

![image](https://user-images.githubusercontent.com/98815410/176678383-3b418ac4-2fdb-435f-98bb-1c01baf35b48.png)


## Usage

Clone the git repo and create a docker using
the instructions in file "create_docker_image.txt"
This will allow you to create a docker image, run a container and start the app.
After this you will have access to the app locally using your webbrowser with:

http://localhost:5000

you can also access a deployed version of the app on Heroku using:

https://concreteinspection.herokuapp.com/info/


## Mission objectives

- Application of deep learning models for the analysis of images.
- Explore Deep Learning libraries: (Tensorflow, PyTorch, etc...).
- Explore other AI approaches that might be useful to identify the cracks and associated geometry.

## The Mission

The client is a construction company using drones to inspect the structures built with concrete. The use of drones allows them to collect a significant amount of pictures of concrete walls, bridges, and concrete decks. The use of drones is helpful because it facilitates the inspection of places with difficult access. However, it also implies the manual inspection of hundreds or thousands of images to identify cracks that might be present on the concrete surface.

It is fundamental for the company to detect even smaller cracks to correct the problem before the appearance of severe structural damage. The next step in the project will be to develop an algorithm that identifies the crack and its length (in mm or cm) to categorize the severity and take immediate action according to it.


### Must-have features

- Detection of cracks on images of concrete.
- Identification of crack severity. This can be quantified by measuring the length of crack.

### Miscellaneous information

The project will require your creativity to combine the different skills that you have acquired to deliver a solution.

Different approaches can be taken depending on what you are aiming to achieve.

- Is there a possibility of using the tool to identify another type of damage on concrete?
- Could the tool be used to identify a crack in another type of material e.g. pavement?

### Data

Different datasets are available from research institutions worldwide, we selected the following dataset:

- [Utah University SDNET2018](https://digitalcommons.usu.edu/all_datasets/48/)


## Technical Evaluation criteria

- Use of deep learning and computer vision algorithms and libraries.
- A baseline model was established.
- The algorithm can detect a crack in concrete.
- The algorithm can estimate the length of the crack.
- Preprocessing of the images was done to improve detection.
- Appropiate metrics were used to evaluate the model.




