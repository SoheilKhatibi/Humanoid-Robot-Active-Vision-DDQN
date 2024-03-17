<a name="readme-top"></a>

<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

<!--[![Contributors][contributors-shield]][contributors-url] -->
<!--[![Forks][forks-shield]][forks-url] -->
<!--[![Stargazers][stars-shield]][stars-url] -->
<!--[![Issues][issues-shield]][issues-url] -->
<!--[![MIT License][license-shield]][license-url] -->
<!-- [![LinkedIn][linkedin-shield]][linkedin-url] -->

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/kOX_vY6ir5M/0.jpg)](https://www.youtube.com/watch?v=kOX_vY6ir5M)

<!-- PROJECT LOGO -->
<br />
<div align="center">
<!--   <a href="https://github.com/SoheilKhatibi/Humanoid-Robot-Active-Vision-DDQN">
    <img src="worlds/Rules_2019.png" alt="Preview" width="1051" height="500">
  </a> -->
  <h3 align="center">Humanoid-Robot-Active-Vision-DDQN</h3>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#Introduction">Introduction</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
<!--         <li><a href="#installation">Installation</a></li> -->
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
<!--     <li><a href="#roadmap">Roadmap</a></li> -->
<!--     <li><a href="#contributing">Contributing</a></li> -->
<!--     <li><a href="#license">License</a></li> -->
    <li><a href="#References">References</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>


# Introduction
This is the first official Code Release of ["Real-time Active Vision for a Humanoid Soccer Robot using Deep Reinforcement Learning"](https://doi.org/10.5220/0010237307420751) paper.

Also, hereby, a world model method is open-sourced. This world model method was introduced by [Meisam Teimouri](https://www.linkedin.com/in/meisam-teimouri-070131222/). It is converted to C++ by me, [Soheil Khatibi](https://www.linkedin.com/in/soheilkhatibi/), in order to reduce the run-time.

Unfortunately, the code, especially the world model section, is too messy. it is to be well documented and cleaned.

### Built With

The project is built using Tensorflow, specifically [Stable Baselines](https://stable-baselines.readthedocs.io/en/master/), and [Webots](https://cyberbotics.com/) Simulation.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

# Getting Started

In order to get started with this project, you need to use [RoboCup-Humanoid-MRL-Webots-Simulation](https://github.com/SoheilKhatibi/RoboCup-Humanoid-MRL-Webots-Simulation), which is the simulation of the [MRL-HSL](https://sites.google.com/view/mrl-hsl) Humanoid Robot in a Kid-Size [Humanoid Soccer League](https://humanoid.robocup.org/) [RoboCup](https://www.robocup.org/) 2019 environment.

# Prerequisites
To run this simulated environment, you need to install:
- [Webots](https://cyberbotics.com/) R2023b. ([Instructions](https://www.cyberbotics.com/doc/guide/installing-webots))
- Tensorflow (GPU) 1.15
- stable-baselines 2.10 ([Link](https://stable-baselines.readthedocs.io/en/master/))
- gym 0.15.7

This code was developed and tested with Python 3.6 (using Miniconda) installed on Ubuntu 20.04

<p align="right">(<a href="#readme-top">back to top</a>)</p>

# Usage

After installing Webots R2023b, You can open this environment as follows:
1. Clone the [RoboCup-Humanoid-MRL-Webots-Simulation](https://github.com/SoheilKhatibi/RoboCup-Humanoid-MRL-Webots-Simulation) repository and run it (instructions could be found [here](https://github.com/SoheilKhatibi/RoboCup-Humanoid-MRL-Webots-Simulation)):
   ```sh
   git clone https://github.com/SoheilKhatibi/RoboCup-Humanoid-MRL-Webots-Simulation.git
   cd RoboCup-Humanoid-MRL-Webots-Simulation
   /path/to/webots worlds/Rules_2019.wbt
   ```
2. In another terminal, clone the repository:
   ```sh
   git clone https://github.com/SoheilKhatibi/Humanoid-Robot-Active-Vision-DDQN.git
   ```
3. Build the world model module:
   ```sh
   cd Humanoid-Robot-Active-Vision-DDQN
   cd RobotWorldModel
   make
   ```
4. Run the Code:
   ```sh
   cd Humanoid-Robot-Active-Vision-DDQN
   cd DQNController
   python DQNController.py
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

# References

- This ReadMe is created using [this](https://github.com/othneildrew/Best-README-Template) useful template, by [othneildrew](https://github.com/othneildrew).
- In the Robot World Model section, The lua parser module for C++ is copied from the code of a blog post which unfortunately I can not find to reference here.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
# Contact

Soheil Khatibi - [Soheil Khatibi](https://www.linkedin.com/in/soheilkhatibi/) - soheilkhatibi1377@gmail.com

Project Link: [https://github.com/SoheilKhatibi/RoboCup-Humanoid-MRL-Webots-Simulation](https://github.com/SoheilKhatibi/RoboCup-Humanoid-MRL-Webots-Simulation)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
# Acknowledgments

Special Thanks to:

* [Meisam Teimouri](https://www.linkedin.com/in/meisam-teimouri-070131222/) for contributing in the project



<p align="right">(<a href="#readme-top">back to top</a>)</p>
