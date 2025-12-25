# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

LeetCUDA 是一个 CUDA 学习和演示项目，包含多个独立的 GPU 计算示例，每个示例都是独立的子项目。

## 构建和运行

### 通用构建命令

每个子项目都有独立的 CMakeLists.txt：

```bash
cd <子项目目录>
mkdir build && cd build
cmake ..
make
```

### 子项目列表

| 目录 | 项目 | 可执行文件 | 说明 |
|------|------|------------|------|
| `CUDAINFO/` | GPU 信息查看 | `cuda_info` | 显示当前 GPU 设备信息 |
| `elementwise/matrixadd/` | 矩阵加法 | `f32add_main`, `f32add_test` | 16×16 线程块的矩阵加法 |
| `elementwise/vectoradd/` | 向量加法 | `f32add` | 256 线程块的向量加法 |

### 运行测试

`matrixadd` 项目包含完整的测试套件：

```bash
cd elementwise/matrixadd/build
./f32add_test    # 运行 9 个测试用例
./f32add_main    # 运行演示程序
```

## 代码架构

### 项目组织模式

每个 CUDA 示例遵循相同的组织结构：

```
<项目名>/
├── CMakeLists.txt      # 构建配置
├── f32add.cuh          # 内核声明和包装函数
├── f32add.cu           # CUDA 内核实现
├── main.cu             # 演示程序（可选）
└── test.cpp            # 测试套件（可选）
```

### CUDA 内核实现模式

**文件分离**：
- `.cuh` - 内核声明和包装函数（`__global__` 和 C++ 接口）
- `.cu` - 实际的 CUDA 内核代码实现
- `.cpp` - CPU 测试代码（可以调用 CUDA 包装函数）

**内核配置**：
- 矩阵运算：使用 16×16 二维线程块
- 向量运算：使用 256 一维线程块
- 网格大小计算公式：`(size + block_size - 1) / block_size`

**内存布局**：
- 矩阵使用行主序（row-major）存储
- 索引计算：`index = row * cols + col`

### CMake 配置要点

- 最低 CMake 版本：3.16
- CUDA 架构支持：75, 80, 86（覆盖 RTX 20/30/40 系列）
- 调试选项：`-g -G -lineinfo`
- C++ 标准：C++17（测试代码）

## 添加新的 CUDA 示例

创建新的子项目时，参考 `elementwise/matrixadd/` 的结构：

1. 创建目录并添加 `CMakeLists.txt`
2. 分离 `.cuh`（声明）和 `.cu`（实现）
3. 添加 `main.cu` 演示程序
4. 添加 `test.cpp` 测试套件（验证 CPU 和 GPU 结果一致）
5. 在测试中使用浮点数容差比较（如 `1e-6`）
