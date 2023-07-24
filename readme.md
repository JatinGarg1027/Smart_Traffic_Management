<h1 align="center">Smart Traffic Management System</h1>

<div align="center">

<h4>The Adaptive Traffic Signal Timer uses live images from the cameras at traffic junctions for traffic density calculation using YOLO object detection and sets the signal timers accordingly, thus reducing the traffic congestion on roads, providing faster transit to people, and reducing fuel consumption.</h4>

</div>

-----------------------------------------
### Inspiration

* Traffic congestion is becoming one of the critical issues with the increasing population and automobiles in cities. Traffic jams not only cause extra delay and stress for the drivers but also increase fuel consumption and air pollution. 

* Current traffic light controllers use a fixed timer and do not adapt according to the real-time traffic on the road.

* This traffic management system act as a traffic light controller that can autonomously adapt to the traffic situation at the traffic signal. The system sets the green signal time according to the traffic density at the signal and ensures that the direction with more traffic is allotted a green signal for a longer duration of time as compared to the direction with lesser traffic. 

------------------------------------------
### Implementation

The project consists of 3 modules:

1. `Vehicle Detection Module` - This module is responsible for detecting the number of vehicles in the image received as input from the camera. More specifically, it will provide as output the number of vehicles of each vehicle class such as car, bike, bus and truck.

<p align="center">
 <img height=200px src="pictures\detection.jpg">
</p>

2. `Signal Switching Algorithm` - This algorithm updates the red, green, and yellow times of all signals. These timers are set bases on the count of vehicles of each class received from the vehicle detection module and several other factors such as the number of lanes, average speed of each class of vehicle, etc. 
<p align="center">
 <img height=200px src="pictures\Algorithm.png">
</p>

3. `Simulation Module` - A simulation is developed using Pygame library to simulate traffic signals and vehicles moving across a traffic intersection.
<p align="center">
 <img height=200px src="pictures\pyg_simu.jpg">
</p>

------------------------------------------
### Demo
<p align="center">
 <img height=200px src="pictures\Modules.png">
</p>

* `Vehicle Detection`

<p align="center">
 <img height=400px src="pictures\vehicle-detection.jpg" alt="Vehicle Detection">
</p>

<br> 

* `Signal Switching Algorithm and Simulation`

<p align="center">
    <img src="pictures\Demo.gif">
</p>