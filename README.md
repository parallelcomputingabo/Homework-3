# Homework 3

By Thai Nguyen.

## How to run

1. Clone this repository
2. Copy the __homework3.ipynb__ notebook
3. Open it in Google Colabs
4. In Google Colabs, go to Runtime -> Change runtime Type
5. Select "T4 GPU" and save
6. Upload __main.cu__ as well as the __data__ folder into the root of the environment
7. The notebook already has commands to run all cases. So just go to Runtime -> Run all

## Explanation and findings

Much of the code is the same as in homework 1 and 2. The exact same functions
are used for matrice extraction and file comparisons. The only differenceb is that
the naive math multiplication has been rewritten with respect to block and threads in the CUDA API.
In homework 2, OpenMP could parallelize multidimensional loops, but CUDA has no such feature. 

Instead, the GPU memory consists of a grid, with several blocks. Each block has multiple threads running in parallel, each handling individual tasks.
The multidimensional loop used previously is running sequentially (no OpenMD), meaning it only runs in one thread. By rewriting that function into a kernel,
and dismissing the multidimensionality, I used CUDA API's to assign each iteration to their own thread which then performs the multiplication. This way, 
the GPU performs multiplication immediately and in parallel (it does not necessarily wait for all threads to be filled, meaning a thread
starts performing as soon as a task is assigned).


| Test Case | Dimensions (m × n × p) | Naive Time (s) | Blocked Time (s) | Parallel Time (s) | Blocked Speedup | Parallel Speedup | Naive CUDA (s) 
|-----------|------------------------|----------------|------------------|-------------------|-----------------|------------------|---------------
| 0         | 64 x 64 x 64           | 0.0012         | 0.0016           | 0.0024            | 0.76×           | 0.50×            | 0.00017
| 1         | 128 x 64 x 128         | 0.0050         | 0.0062           | 0.0055            | 0.81×           | 1.01×            | 0.00020
| 2         | 100 x 128 x 56         | 0.0034         | 0.0042           | 0.0040            | 0.81×           | 0.87×            | 0.00023
| 3         | 128x64x128             | 0.0050         | 0.0066           | 0.0047            | 0.76×           | 1.06×            | 0.00017
| 4         | 32x128x32              | 0.0006         | 0.0007           | 0.0019            | 0.83×           | 0.33×            | 0.00024
| 5         | 200x100x256            | 0.0243         | 0.0327           | 0.0151            | 0.76×           | 1.64×            | 0.00040
| 6         | 256x256x256            | 0.0878         | 0.1070           | 0.0512            | 0.82×           | 1.71×            | 0.00062
| 7         | 256x300x256            | 0.1051         | 0.1251           | 0.0456            | 0.84×           | 2.30×            | 0.00065
| 8         | 64x128x64              | 0.0025         | 0.0032           | 0.0028            | 0.70×           | 0.86×            | 0.00025
| 9         | 256x256x257            | 0.0826         | 0.1041           | 0.0484            | 0.79×           | 1.70×            | 0.00062
| 10        | 512x512x512            | 0.9335         | 0.8905           | 0.2717            | 1.05×           | 3.43×            | 0.00034
| 11        | 1024x1024x1024         | 7.7868         | 5.3085           | 0.9058            | 1.46×           | 8.59×            | 0.02512

Due to homework 1 and 2 being written on my own machine (which does not have Nvidia GPU), this Homework
was written and used in Google Colabs. I can't really do an unbiased comparisons
between the results from this assignment with the previous ones, due to possible performance differences
between my machine and Google hardwares. But, in this situation, the output using CUDA far exceeds what my
Intel i5 2500 is capable of.


## Computer specs:

- Google Colabs
- Hardware Accelerator: T4 GPU