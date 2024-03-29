# conveyor_belt_pizza_counting_system

https://github.com/umairsiddique3171/conveyor_belt_pizza_counting_system/assets/148565997/a06f43f5-d435-4b2b-a9bd-83db5c8a325e

## Overview
This project implements a conveyor belt pizza counting system using computer vision techniques. The system accurately counts the number of pizzas passing down conveyor belts in a camera footage. 

This was one of the projects I did on Fiverr. 

**Client asked:** I would like the freelancer to code a script in Python that will accurately count the number of pizzas passing down conveyor belts. After the video is finished, the final count should be saved in a variable.

## Project Workflow 
- **Data Extraction:** Training images were extracted from camera footage to train the classifier. Video frames were sampled to capture various instances of pizzas on the conveyor belt, providing diverse data for classifier training.
- **Classifier Training:** A classifier was trained using machine learning techniques to detect pizzas within the frames.
- **Region of Interest (ROI) Definition:** Specific regions of interest (ROI) were defined within each frame where pizza detection would be performed.
- **Pizza Detection:** The trained classifier was applied to the ROI to detect the presence of pizzas.
- **Counting Logic:** Pizza counts were incremented based on detections within the ROI.
- **Displaying Results:** The count was displayed on each frame for visual verification.
- **Output Generation:** Processed frames with count information were compiled into an output video file (results.mp4).


## Future Improvements
- Enhance detection accuracy through advanced computer vision algorithms.
- Implement real-time counting for live video streams.
- Develop a user-friendly interface for easier configuration and usage.

## License
This project is licensed under the [MIT License](https://github.com/umairsiddique3171/conveyor_belt_pizza_counting_system/blob/main/LICENSE)

## Author 
[@umairsiddique3171](https://github.com/umairsiddique3171)
