# Project 2: Content-based Image Retrieval

The purpose of this project is to continue the process of learning how to manipulate and analyze images at a pixel level. In addition, this is the first project where we will be doing matching, or pattern recognition.

## Files

* main.cpp: Support all feature algorithms.
* main_part2.cpp: Support part2 of baseline matching.
* include/feature.h: Implement various CBIR features.
* include/distance.h: Implement various distance metrics.
* include/csv_util.h: Implement functions for csv handling.

Suported feature types: baseline, histogram, multihistogram, dnn, texture, gabor, grass, bluebins, select ROI.

For main.cpp
```
./project2_app <directory path> <target image path> <feature type> <n> <dnn feature file (optional)> <select ROI boolean (optional)>
```

For main_part2.cpp:
```
# Writing CSV: Usage: 
./project2_app <directory path> <output CSV file>
# Comparing images: Usage: 
./project2_app <target image> <feature vector file> <N>
```
