Test Case	Dimensions (m × n × p)	Naive CPU	Blocked CPU	Parallel CPU	Naive CUDA	Tiled CUDA	Speedup (Tiled vs Naive CUDA)	Speedup (Tiled CUDA vs Parallel CPU)
0	64 × 64 × 64	0.00287	0.00398	0.00398	0.10490	0.08358	1.26×	0.05×
1	128 × 64 × 128	0.00820	0.01073	0.00162	0.14189	0.09427	1.50×	0.02×
2	100 × 128 × 56	0.00592	0.00705	0.00257	0.12102	0.08058	1.50×	0.03×
3	128 × 64 × 128	0.00855	0.01046	0.00159	0.13971	0.09542	1.46×	0.02×
4	32 × 128 × 32	0.00102	0.00143	0.00062	0.12237	0.06342	1.93×	0.01×
5	200 × 100 × 256	0.03097	0.02394	0.00744	0.21674	0.17594	1.23×	0.04×
6	256 × 256 × 256	0.05747	0.08263	0.02420	0.31066	0.24397	1.27×	0.10×
7	256 × 300 × 256	0.06370	0.09229	0.01788	0.29862	0.24682	1.21×	0.07×
8	64 × 128 × 64	0.00379	0.00581	0.00122	0.12451	0.07613	1.64×	0.02×
9	256 × 256 × 257	0.05919	0.07627	0.02289	0.29376	0.24883	1.18×


Based on the timing data collected for all ten test cases, I can see a clear trend: the tiled CUDA implementation consistently outperforms the naive CUDA version, but it still doesn’t quite beat my CPU’s parallel implementation once data transfers are included.

For each case, I measured wall-clock times (including host→device and device→host transfers) for:
Naive CUDA
Tiled CUDA
My Assignment 2 implementations (naive, blocked, and OpenMP-parallel CPU)

Run on CSC:
module load cuda
module load GCC
module load cmake
nvcc -arch=sm_70 main.cu -o mul_m -lm
srun -p gpu --mem=10G -t 1:00:00 ./mul_m <test_case>