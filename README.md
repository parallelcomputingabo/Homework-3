


#### --3. Performance Measurements--
I landed on a tile_width of 16 and not 32. Probably the point was to test out some other values as well.
Though once more I am being honest... I didn't do too much testing. I want to go sleep.


| Test Case | Dimensions (\( m \times n \times p \)) | Naive CPU (ms) | Blocked CPU (ms) | Parallel CPU (ms) | Naive CUDA (ms) | Tiled CUDA (ms) | Tiled CUDA Speedup (vs. Naive CUDA) | Tiled CUDA Speedup (vs. Parallel CPU) |
|-----------|--------------------------------|----------------|------------------|-------------------|--------------|-----------------|-------------------------------------|---------------------------------------|
| 0         | 64x64x64                       | 0.7018         | 0.6433           |    0.8423               | 1.9768             |0.1141               | ~17x                                | ~7x                                   |
| 1         | 128x64x128                       |       2.7944         | 2.5573           |     1.0688              |    1.7854          |        0.1943      | ~9                     x            | ~6                             x      |
| 2         | 100x128x56                      |    1.929            | 1.7388           |      0.9698             |    1.8416          |       0.1815          | ~      10              x            | ~5                            x       |
| 3         | 128x64x128                   |       2.8365         | 2.556            | 1.1706                  |        1.9257      |          0.152     | ~                   13       x      | ~                   8              x  |
| 4         | 32x128x32                              |        0.3528        |         0.3139         |      0.7155             |   1.7968           |   0.0899            | ~        20  x                      | ~       8         x                   |
| 5         | 200x100x256                        |      13.9896         |       12.5345           |          2.6496          |     1.7825         |      0.2821        | ~                     6     x       | ~                9   x                |
| 6         | 256x256x256                        |    44.9417            |   40.3454               |       6.1479            |        3.0656      |      0.308         | ~                    10      x      | ~20                 x                 |
| 7         | 256x300x256                        |       52.6618         |              47.4069    |      7.9057             |        1.8641      |           0.3371   | ~                    5        x     | ~              23  x                  |
| 8         | 64x128x64                      |       1.4304         |          1.2571        |         0.9347          |     1.7844         |           0.1202      | ~        15                      x  | ~                   8x                |
| 9         | 256x256x257                        |     45.4665           |       40.6735           |          6.2421         |      1.86        |         0.311        | ~                         6    x    | ~             20x                     |
I do wonder how exactly I have screwed up these implementations for osmeo of the end edcases to be so different. Though for 7 and 9 I guess the datasets are quite large, thus cpu doesn't enjoy them.


### Build Instructions

- Use the provided `CMakeLists.txt`, which includes CUDA support.
- **Requirements**:
    - NVIDIA GPU with CUDA support.
    - CUDA Toolkit installed (version 11.x or later recommended).
    - CMake with CUDA language support.
- **Linux/Mac**:
    - Run `cmake -DCMAKE_CUDA_COMPILER=nvcc .` to generate a Makefile, then `make`.
- **Windows**:
    - Use Visual Studio with CUDA toolkit or MinGW with `cmake -G "MinGW Makefiles"`.
- Test with the same test cases (0–9) as Assignment 2.

---