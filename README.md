# CS5330ComputerVision
CS5330 Pattern Recognition and Computer Vision

To build and run a particular project,

Create a new build directory inside the project and excute the binary:

```
mkdir build
cd build

cmake ..
make
./OpenCVTest
```

---

To simple test program, you can use direct method (note that you need to specify all dependent files and lib):

```
g++ -std=c++11 main.cpp -o app `pkg-config --cflags --libs opencv4`
```
