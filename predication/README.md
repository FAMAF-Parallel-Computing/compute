# Predication AVX512

```bash
clang++ main.cpp predication.cpp -O3 -march=znver5 -std=c++23 -o predication
```

```bash
clang++ predication.cpp -O3 -march=znver5 -std=c++23 -S -o - | llvm-mca -mcpu=znver5 -timeline
```
