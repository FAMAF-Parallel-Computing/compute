# Stencil

## Analysis

### Build for AVX512

```bash
clang++ kernels.cpp -march=znver5 -O3 -o kernels.S
# annotate the assembly...
llvm-mca -mcpu=znver5 -timeline kernels.S -o kernels.mca
```
## Filter

To only execute some benchmarks execute 
`--benchmark_filter=<regex>`

## References

> Sharma, S. (2019). Multi-Dimensional Auto-Vectorization of Stencil Codes (Doctoral dissertation, Saarland University).
