# Chair Imagination
__Is That a Chair? Imagining Affordances Using Simulations of an Articulated Human Body (ICRA2020)__

Hongtao Wu, Deven Misra, [Gregory Chirikjian](https://me.jhu.edu/faculty/gregory-s-chirikjian/)

* [Paper on IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/9197384)
* [Paper on arxiv](https://arxiv.org/pdf/1909.07572.pdf)
* [Project Page & Video Results](https://chirikjianlab.github.io/chairimagination/)
* [Data](https://www.dropbox.com/s/fpnxhigttq06w1w/contain_imagine_data_RAL2021.zip?dl=0)


## Installation
- Pybullet (3.0.8): Simulation platform
```
pip install pybullet==3.0.8
```
- Trimesh (3.9.1): Used for measure geometry of the object
```
pip install trimesh==3.9.1
```
- Miscellaneous
```
pip install numpy==1.16.6
```
- V-HACD:  Build the library from the [V-HACD repo](https://github.com/kmammou/v-hacd)
    ```
    git clone https://github.com/kmammou/v-hacd
    cd v-hacd/src/
    mkdir build && cd build
    cmake ..
    make

- [meshlab (2020.03)](https://github.com/cnr-isti-vclab/meshlab/tree/f3568e75c9aed6da8bb105a1c8ac7ebbe00e4536): Used for getting the physical information of the scanned object. Latest version of meshlab will not work with xml file. It needs python 3.6 (Pymeshlab) to interface with. Download the source code from the [Release](https://github.com/cnr-isti-vclab/meshlab/tags) page. Clone the vcglib after unzip the code. Remember to checkout to the correct commit meshlab 2020.03 was using.
    ```
    cd <PATH_TO_MESHLAB>
    git clone https://github.com/cnr-isti-vclab/vcglib.git
    cd vcglib
    git checkout 5fa560e9e6
    ```
    Then go to the root directory and use the bash file in /install to install.
    ```
    cd ../
    bash install/linux/linux_setup_env_ubuntu.sh
    bash install/linux/linux_make_it.sh
    ```
    The executable of **meshlab** and **meshlabserver** will be installed in /distrib. We will be using the meshlabserver in the code to compute geometric metrics.


## Usage
- Specify the data folder path, result folder, v-hacd path and meshlab path in ```./src/chair_imagination_clasification.py```

- Run the experiment.
```python src/chair_imagination_clasification.py```


## Citation
If you find this code and/or the data useful in your work, please consider citing
```
@inproceedings{wu2020chair,
  title={Is that a chair? imagining affordances using simulations of an articulated human body},
  author={Wu, Hongtao and Misra, Deven and Chirikjian, Gregory S},
  booktitle={2020 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={7240--7246},
  year={2020},
  organization={IEEE}
}
```
