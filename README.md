**Parallel Programming**  
**Åbo Akademi University, Information Technology Department**  
**Instructor: Alireza Olama**

**Homework Assignment 3: Matrix Multiplication with CUDA**

**Due Date**: **31/05/2025**  
**Points**: 100

---

| Test Case | Dimensions (\( m \times n \times p \)) | Naive CPU (s) | Blocked CPU (s) | Parallel CPU (s) | Naive CUDA (s) | Tiled CUDA (s) | Tiled CUDA Speedup (vs. Naive CUDA) | Tiled CUDA Speedup (vs. Parallel CPU) |
|-----------|----------------------------------------|---------------|-----------------|------------------|----------------|----------------|-------------------------------------|---------------------------------------|
|     0    |            64x64x64             |   0   |     0      |      0       |     0.15     |    0.135      |              1.11                 |      0                          |
|     1    |            128x64x128             |   0.00199986   |     0.00200009      |      0.000999928       |    0.18      |     0.2     |           0.9                    |           0                     | 
|     2    |            100x128x56             |   0.00199986   |      0.00200009     |     0.00100017        |     0.15     |    0.172      |           0.87                    |              0                   |
|     3    |            128x64x128             |   0.00300002   |      0.00199986     |      0.000999928       |     0.167     |     0.193     |          0.86                     |             0                    |
|     4    |            32x128x32             |   0   |     0      |     0        |     0.125     |    0.113      |            0.904                   |      0                           |
|     5    |            200x100x256             |  0.0120001    |      0.0109999     |      0.00399995       |    0.279      |    0.473      |              0                 | 0                                |
|     6    |            256x256x256             |   0.036   |      0.0419998     |      0.00600004       |     0.385     |    0.88      |       0.589                        |  0                               |
|     7    |            256x300x256             |   0.0409999   |     0.043      |      0.00999999       |     0.386     |     0.982     |       0.393                        |  0                               |
|     8    |            64x128x64             |   0.000999928   |     0.00200009      |      0.000999928       |    0.141      |   0.164       |    0.859                           |                 0                |
|     9    |            256x256x257             |   0.0319998   |      0.036     |      0.0079999       |      0.369    |    0.905      |      0.407                         |  0                               |

---
## Some extra explanation about the assignment
### Files
Output of the calculations nodes as a text, executable, cuda source file and the bash file for jub run.
### Tiled implementation
The implementation of the the tiled multiplication has been borrowed from [Here](https://medium.com/@dhanushg295/mastering-cuda-matrix-multiplication-an-introduction-to-shared-memory-tile-memory-coalescing-and-d7979499b9c5). Two values (16 and 32) has been tested for the TILE_WIDTH, however, 32 didn't work and was giving wrong answers, it may be to large for the shared memory.
### Transfer time
Transfer time has been calculated seperately and added to the calculation time. It included time to copy from host to device and vice versa. The seperated times are showed in the text output.
### Timings and the table
I don't know why but my naive is faster in almost all of the scenarios. Even though I checked the implementation with other resources.
The timings of assignment2, seem much better, I don't know what is the reason for that, it might be an issue with the timer in one of the versions but due to this difference, speed up of GPU was meaningless and close to zero for all of the cases.


