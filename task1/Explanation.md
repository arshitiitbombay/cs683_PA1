# Explanation

## Task 1

### conv_reorder

* **Initially**, we calculated each `out[i]` completely by looping over all \(K\times K\) kernel elements and accumulating their contributions. **After reordering**, we instead fix a kernel element \(K[ky][kx]\) and apply its contribution to all relevant output elements before moving to the next kernel element. Mathematically, the original approach is:

$$
out[oy][ox] =
\sum_{ky=0}^{K-1}\sum_{kx=0}^{K-1}
in[oy+ky][ox+kx]\cdot K[ky][kx]
$$

whereas the reordered approach computes:

$$
out[oy][ox] \mathrel{+}=
in[oy+ky][ox+kx]\cdot K[ky][kx]
$$

for each \(K[ky][kx]\) across the output before proceeding to the next kernel element.

- This is better as in the first there is less locality. In the first we change rows againa and again in each loop ( kernel one ) , but in the second we go linearly in each loop, making better access and giving more locality. This might also not work coz we are touching the L1D cache again and again, 9x more. Results say that. Associativity does a good job here and helps the naive out here.


### conv_unroll

- Loop unrolling make the instructions independent ( atleast in the last 2 loops ) and amke the superscalar make use of that independence, amking the work faster.

- Also another point I learnt is that when we do :
    const float k00 = ker[0];
  it hints the compiler to use the registers and if we have small number of values , it might utilize that to make it even faster. If we were given a fixed kernel size , we could impliment that also to increase the speed.

### conv_tile

- System details :
NAME ONE-SIZE ALL-SIZE WAYS TYPE        LEVEL  SETS PHY-LINE COHERENCY-SIZE
L1d       32K     256K    8 Data            1    64        1             64
L1i       32K     256K    8 Instruction     1    64        1             64
L2       256K       2M    4 Unified         2  1024        1             64
L3        16M      16M   16 Unified         3 16384        1             64

- Tiling helps in localising the cache to the area that will be used again and again. Mathematically , shown in class.

For a 32 KB L1-D cache:

- Cache size = `32 × 1024 = 32,768 bytes`
- Each `float` = `4 bytes`
- Therefore, cache can hold approximately:

  `32,768 / 4 = 8,192 floats`

For a `T × T` tile with a `3 × 3` kernel, we need approximately:

`(T + 2)²` input floats + `T²` output floats.

For `T = 32`:

`34² + 32² = 2,180 floats`

`2,180 × 4 ≈ 8.5 KB`

Thus, a `32 × 32` tile comfortably fits within the 32 KB L1-D cache.

- Experimented wiht multiple sizes , but miss rate is high ( >5% ) 

- (Arshit) As of now I think that the hardware prefetchers are helping the naive version and hindering the tiled version

- Naive LDMPKI was around 2 ( We have whole graphs and file for the information )

- (Arshit) Tile size 32 gives good results for me ( around 1.1x )

- Currently L1d- MPKI is increased moving to tiled from naive which is counter intuitive 

- (Arshit )As tile size increase , speedup decreased. This I think is because of limit of the cache. We need ( T + P -1)^2 + K^2 elements , which should fit in cache for around 48 tile size also but that I think is the main point, cache being over the brim. 

- (Arshit )A very little speedup of 1.1x ( on K = 3 ) and 1.14x ( K = 5 ). I'm not sure why the speedup is even there even though speedup is max for 32 bit and ldmki is least for 64 bit

### conv_simd

- Naive   : 0.057784 s    610574474 instructions
- 128-bit : 0.032623 s    265584810 instructions
- 256-bit : 0.038727 s    257196196 instructions

- As size is increased , the number of instructions grows in both but to a lesser extent in simd. 

- As we increase the size , the speedup increases. Reason : The elements are 

- simd intrinsics used :
   _mm256_fmadd_ps : acc += k * val ; good rather than multiplying and adding , a single hardware intr might be better

### conv_optimized

- Graphs made for some techniques
  Naive
  Tiling
  Unrolling
  Rodering
  Final Optimised

- Final OPtimised works like this :

For K =3 ( which is to be graded , right ? : | ) , we do unrolling + simd + hint to store kernel in AVX2 registers , while for others we do simple unrolling by 4 and simd. 

- Graphs indicate high increment when k=3. For k=5 and 7 better than simd by +2.0x ( around )

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------