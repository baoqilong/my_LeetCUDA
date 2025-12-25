## Matrix Addition

Implement a program that performs element-wise addition of two M × N matrices containing 32-bit floating point numbers on a GPU. The program should take two input matrices of equal dimensions and produce a single output matrix containing their element-wise sum.

## Implementation Details

### Kernel Configuration
- **Thread Block**: 16 × 16 threads (256 threads per block)
- **Grid**: Calculated based on matrix dimensions
  - Grid X = `(cols + 15) / 16`
  - Grid Y = `(rows + 15) / 16`

### Memory Layout
Matrices are stored in **row-major** order:
```
Index = row * cols + col
```

## Files

| File | Description |
|------|-------------|
| [f32add.cuh](f32add.cuh) | Header with kernel declarations |
| [f32add.cu](f32add.cu) | CUDA kernel implementation |
| [main.cu](main.cu) | Demo program |
| [test.cpp](test.cpp) | Test suite |
| [CMakeLists.txt](CMakeLists.txt) | Build configuration |

## Build & Run

```bash
cd build
cmake ..
make
./f32add_main   # Run demo
./f32add_test   # Run tests
```

## Examples

**Example 1:**
```
Input:  A = [[1.0, 2.0],
             [3.0, 4.0]]

        B = [[5.0, 6.0],
             [7.0, 8.0]]

Output: C = [[6.0, 8.0],
             [10.0, 12.0]]
```

**Example 2:**
```
Input:  A = [[1.5, 2.5],
             [4.5, 5.5],
             [7.5, 8.5]]

        B = [[0.5, 0.5],
             [0.5, 0.5],
             [0.5, 0.5]]

Output: C = [[2.0, 3.0],
             [5.0, 6.0],
             [8.0, 9.0]]
```

## Test Cases

| Test Case | Dimensions | Input A | Input B | Description |
|-----------|------------|---------|---------|-------------|
| Basic test | 12 × 10 | 1.0 | 1.0 | Original demo case |
| Exact block | 16 × 16 | 2.5 | 3.7 | Matches block size |
| Single element | 1 × 1 | 1.0 | 2.0 | Minimum size |
| Large square (zeros) | 100 × 100 | 0.0 | 0.0 | Edge case: zeros |
| Large square (negation) | 100 × 100 | 1.0 | -1.0 | Edge case: cancellation |
| Wide matrix | 5 × 200 | 3.14 | 2.86 | Non-square (wide) |
| Tall matrix | 200 × 5 | 1.0 | 1.0 | Non-square (tall) |
| Non-multiple of 16 | 17 × 33 | 1.5 | 2.5 | Partial blocks |
| Very large | 1024 × 1024 | 1.0 | 1.0 | Stress test |