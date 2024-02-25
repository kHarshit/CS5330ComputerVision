# Project 2: Content-based Image Retrieval

The project aimed to identify shapes of objects, in real time. It uses binary image processing techniques to improve the clarity of images and then separates the main objects from the background. By analyzing key features of these objects, the system gathers information to categorize them into different groups. The objects in the video frames were classified based on their similarity to existing categories in the database based on their features. To evaluate the system's effectiveness, we use a confusion matrix through which accuracy can be calculated.

## How to run?

```
# compile
cmake ..
make
# run
./project3_app
```

### System Info

System (OpenCV4 with VSCode): 
```
Darwin Harshits-MacBook-Pro.local 23.2.0 Darwin Kernel Version 23.2.0: Wed Nov 15 21:54:55 PST 2023; root:xnu-10002.61.3~2/RELEASE_ARM64_T8122 arm64
```