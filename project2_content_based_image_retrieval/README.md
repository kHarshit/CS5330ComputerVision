# Project 2: Content-based Image Retrieval

The purpose of this project is to continue the process of learning how to manipulate and analyze images at a pixel level. In addition, this is the first project where we will be doing matching, or pattern recognition.

## Files

* main.cpp: Support all feature algorithms.
* main_part2.cpp: Support part2 of baseline matching.
* include/feature.h: Implement various CBIR features.
* include/distance.h: Implement various distance metrics.
* include/csv_util.h: Implement functions for csv handling.

Suported feature types: baseline, histogram, multihistogram, dnn, texture, gabor, grass, bluebins, select ROI.

## How to run?

For main.cpp (include main.cpp in CMakeLists.cpp)
```
# compile
cmake ..
make
# run
./project2_app <directory path> <target image path> <feature type> <n> <dnn feature file (optional)> <select ROI boolean (optional)>
```

For main_part2.cpp (replace main.cpp with main_part2.cpp):
```
# compile
cmake ..
make
# Writing CSV: Usage: 
./project2_app <directory path> <output CSV file>
# Comparing images: Usage: 
./project2_app <target image> <feature vector file> <N>
```

### System Info

System (OpenCV4 with VSCode): 
```
Darwin Harshits-MacBook-Pro.local 23.2.0 Darwin Kernel Version 23.2.0: Wed Nov 15 21:54:55 PST 2023; root:xnu-10002.61.3~2/RELEASE_ARM64_T8122 arm64
```