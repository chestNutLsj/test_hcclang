# HCCLang - 华为HCCL集合通信算法领域特定语言

HCCLang是一个专为华为集合通信库(HCCL)设计的领域特定语言(DSL)，用于定义和生成高效的集合通信算法。该项目基于Microsoft的MSCCLang进行适配开发，为HCCL环境提供了完整的算法描述、优化和代码生成解决方案。

## 项目概述

HCCLang提供了一套完整的工具链，让开发者能够以高级、直观的方式描述集合通信算法，并自动生成针对HCCL运行时优化的实现代码。该项目的核心价值在于弥合算法设计与硬件实现之间的鸿沟，使研究人员和工程师能够专注于算法逻辑本身，而无需深入底层实现细节。

### 核心特性

HCCLang系统具备以下关键特性：**表达能力强大的DSL语法**，支持基于数据块(chunk)的直观算法描述；**HCCL原生支持**，生成与HCCL运行时完全兼容的JSON中间表示；**算法验证机制**，内置正确性检查和性能分析工具；**可视化支持**，能够生成通信模式的图形化表示；**模块化架构**，支持算法组合和分层优化策略。

### 设计理念

HCCLang的设计遵循分离关注点的原则，将算法描述、优化策略和代码生成解耦。算法设计者可以使用高级抽象描述通信模式，优化专家可以应用各种变换和组合策略，而系统工程师则可以专注于运行时集成和性能调优。这种分层设计不仅提高了开发效率，也增强了系统的可维护性和扩展性。

## 系统架构

HCCLang采用了双层次的编译流水线架构。该架构的核心优势在于其**渐进式抽象**特性。开发者从高级DSL开始，逐步向下细化到具体的硬件实现。每一层都保持了足够的抽象性，使得算法设计可以独立于特定硬件平台进行，同时又提供了足够的控制粒度以实现最优性能。

## 模块结构详解

HCCLang采用功能导向的模块化设计，将系统划分为以下核心模块：

### 核心层 (hcclang/core/)

核心层包含算法描述的基础数据结构和集合通信操作的定义。`algorithm.py`定义了Algorithm和Step类，提供了算法表示的基本框架；`collectives.py`包含AllReduce、AllGather、AllToAll等标准集合通信操作的抽象定义。这一层的设计遵循了面向对象的设计原则，为上层模块提供了稳定的编程接口。

### 语言层 (hcclang/language/)

语言层实现了HCCLang DSL的核心语法构造，包括缓冲区管理、数据块操作、中间表示(IR)定义以及编译器优化过程(passes)。该层的设计参考了现代编译器理论，采用了多阶段的IR变换策略，确保了语言的表达能力和优化空间。

### 拓扑层 (hcclang/topologies/)

拓扑层定义了各种硬件网络结构，包括通用拓扑(generic.py)、NVIDIA平台拓扑(nvidia.py)以及华为昇腾平台的CM384拓扑。每个拓扑定义不仅包含节点连接关系，还包含了详细的带宽和延迟建模，为算法优化提供了硬件感知的基础。

### 运行时层 (hcclang/runtime/)

运行时层负责将高级算法描述转换为可执行格式。`hcclize.py`专门针对HCCL运行时生成.h/.cc格式的代码；`ncclize.py`保持了与MSCCL XML格式的兼容性；`serialization.py`提供了通用的序列化工具。这种设计使得HCCLang能够同时支持多种不同的运行时环境。

### 优化层 (hcclang/optimization/)

优化层包含了多种算法优化和组合策略。`composers.py`实现了算法组合器，支持将简单算法组合成复杂的通信模式；`distributors/`子模块专门处理分层通信策略，针对多节点GPU集群进行优化；`ncd_reduction.py`实现了网络编码降维技术，减少通信复杂度。

### 求解器层 (hcclang/solver/)

求解器层提供了自动化算法生成和优化的工具。`instance.py`定义了优化问题的实例表示；`path_encoding.py`实现了通信路径的编码算法；`rounds_bound.py`和 `steps_bound.py`分别计算通信轮次和步骤的理论界限，为算法设计提供理论指导。

### 程序库 (hcclang/programs/)

程序库包含了预定义的标准算法实现，涵盖了常见的集合通信模式。这些实现不仅可以直接使用，也可以作为学习和开发新算法的参考模板。

### 命令行工具 (hcclang/cli/)

命令行工具提供了便捷的脚本化接口，支持批量算法生成、性能测试和结果分析等功能。

## 安装与环境配置

HCCLang的安装过程设计得简单直接，支持多种Python环境管理方案。推荐使用Conda进行环境管理，以确保依赖包的版本兼容性。

### 环境准备

首先创建并激活一个专用的Python环境：

```bash
# 使用Conda创建Python 3.12环境
conda create -n hcclang python=3.12
conda activate hcclang
```

### 依赖安装

HCCLang的依赖包经过精心选择，确保了系统的稳定性和性能：

```bash
# 安装Python依赖
pip install -r requirements.txt

# 开发模式安装HCCLang
pip install -e .
```

### 验证安装

安装完成后，可以通过以下方式验证系统的正常工作：

```bash
# 检查HCCLang模块导入
python -c "import hcclang; print('HCCLang安装成功')"

# 运行基础测试
python -m hcclang --help
```

## 快速入门指南

HCCLang的学习曲线被设计得相对平缓，开发者可以从简单的示例开始，逐步掌握高级特性。

### 基础算法示例

以下是一个完整的Ring AllReduce算法实现示例，展示了HCCLang的基本使用模式：

```python
import hcclang
from hcclang.language import *
from hcclang.topologies import ring
from hcclang.core import allreduce
from hcclang.runtime.hcclize import save_hccl_algorithm

# 定义4节点环形拓扑
topology = ring(4)

# 创建AllReduce集合通信操作
collective = allreduce(4)

# 定义算法步骤
def create_ring_allreduce_steps():
    steps = []
  
    # 第一阶段：Reduce-Scatter
    for step in range(3):
        for rank in range(4):
            src_rank = rank
            dst_rank = (rank + 1) % 4
            chunk_id = (rank - step - 1) % 4
      
            steps.append({
                'rank': src_rank,
                'sends': [{'dst_rank': dst_rank, 'chunk_id': chunk_id}],
                'receives': [{'src_rank': (rank - 1) % 4, 'chunk_id': chunk_id}],
                'reduces': [{'chunk_id': chunk_id}] if step > 0 else []
            })
  
    # 第二阶段：All-Gather
    for step in range(3):
        for rank in range(4):
            src_rank = rank
            dst_rank = (rank + 1) % 4
            chunk_id = (rank - step) % 4
      
            steps.append({
                'rank': src_rank,
                'sends': [{'dst_rank': dst_rank, 'chunk_id': chunk_id}],
                'receives': [{'src_rank': (rank - 1) % 4, 'chunk_id': chunk_id}]
            })
  
    return steps

# 创建算法实例
algorithm = Algorithm.make_implementation(
    topology=topology,
    collective=collective,
    steps=create_ring_allreduce_steps()
)

# 配置生成参数
settings = {
    "ranks": 4,
    "chunks_per_rank": 4,
    "chunk_size_bytes": 1048576,
    "data_type": "HCCL_DATA_TYPE_FP32",
    "reduce_op": "HCCL_REDUCE_SUM"
}

# 生成HCCL JSON输出
save_hccl_algorithm(algorithm, settings, "ring_allreduce.json")
```

### CM384拓扑使用示例

HCCLang对华为昇腾CM384平台提供了专门的支持，以下示例展示了如何在CM384拓扑上定义算法：

```python
from hcclang.topologies.CM384 import CM384_128_slice, CM384_full

# 使用128个NPU的切片配置
topology_128 = CM384_128_slice()
print(f"128-NPU拓扑：{topology_128.num_nodes()}个节点")

# 使用完整的384个NPU配置
topology_384 = CM384_full()
print(f"384-NPU拓扑：{topology_384.num_nodes()}个节点")

# 查看节点间连接带宽
intra_node_bw = topology_128.link(0, 1)  # 节点内UB连接：392 GB/s
inter_node_bw = topology_128.link(0, 8)  # 节点间RDMA连接：400 GB/s
```

## 文档和教程使用指南

HCCLang提供了丰富的学习资源，其中最重要的是交互式Jupyter教程。

### 交互式教程

`docs/ring_allreduce_tutorial.ipynb`是一个完整的交互式教程，涵盖了从算法理论到具体实现的全过程。该教程的特点包括：

**理论基础**：详细解释了Ring AllReduce算法的数学原理和通信模式，包括reduce-scatter和all-gather两个阶段的详细分析。

**可视化展示**：通过图形化的方式展示4个节点在环形拓扑中的数据流动过程，帮助读者直观理解算法执行过程。

**代码实践**：提供了完整的可执行代码，读者可以逐步运行每个代码单元，观察输出结果。

**性能分析**：包含了算法复杂度分析和性能优化建议，帮助读者理解不同设计选择的影响。

### 使用教程的步骤

启动Jupyter环境：

```bash
# 确保在hcclang环境中
conda activate hcclang

# 启动JupyterLab
jupyter lab docs/ring_allreduce_tutorial.ipynb
```

教程按照渐进式的结构组织，建议按顺序学习：

1. **算法背景**：理解Ring AllReduce的基本概念
2. **拓扑定义**：学习如何定义和配置网络拓扑
3. **步骤构造**：掌握算法步骤的详细定义方法
4. **代码生成**：了解JSON输出格式和HCCL集成过程
5. **验证分析**：学习如何验证算法正确性和分析性能

## HCCLang模块使用详解

HCCLang的模块化设计使得开发者可以根据需要选择合适的组件，以下是各模块的详细使用说明。

### 核心模块使用

核心模块提供了算法描述的基础设施：

```python
from hcclang.core import Algorithm, Step
from hcclang.core.collectives import allreduce, allgather, alltoall

# 创建算法实例
algorithm = Algorithm(name="MyAlgorithm", topology=my_topology)

# 定义通信步骤
step = Step(
    rank=0,
    sends=[Send(dst=1, buffer="output", chunk=0)],
    receives=[Recv(src=3, buffer="input", chunk=1)],
    reduces=[Reduce(buffer="temp", chunk=2)]
)

# 添加步骤到算法
algorithm.add_step(step)
```

### 拓扑模块使用

拓扑模块支持多种网络结构的定义：

```python
from hcclang.topologies import ring, tree, mesh
from hcclang.topologies.CM384 import CM384_128_slice

# 标准拓扑
ring_topo = ring(8)           # 8节点环形
tree_topo = tree(16)          # 16节点树形
mesh_topo = mesh(4, 4)        # 4x4网格

# 专用拓扑
cm384_topo = CM384_128_slice() # CM384 128-NPU配置

# 查看拓扑属性
print(f"节点数量：{ring_topo.num_nodes()}")
print(f"连接带宽：{ring_topo.link(0, 1)} GB/s")
```

### 优化模块使用

优化模块提供了多种算法优化策略：

```python
from hcclang.optimization.composers import compose_algorithms
from hcclang.optimization.distributors import hierarchical_alltoall

# 算法组合
combined_algo = compose_algorithms([algo1, algo2, algo3])

# 分层优化（针对多节点场景）
optimized_algo = hierarchical_alltoall(
    topology=multi_node_topo,
    intra_node_algo="ring",
    inter_node_algo="tree"
)
```

### 运行时代码生成

运行时模块负责生成可执行代码：

```python
from hcclang.runtime.hcclize import hcclize, save_hccl_algorithm
from hcclang.runtime.ncclize import ncclize  # NCCL兼容性

# 生成HCCL JSON
json_output = hcclize(algorithm, settings)

# 保存到文件
save_hccl_algorithm(algorithm, settings, "my_algorithm.json")

# 也可以生成NCCL XML（向后兼容）
xml_output = ncclize(algorithm, settings)
```

## 高级特性和最佳实践

HCCLang提供了多种高级特性，帮助开发者创建更加高效和复杂的算法。

### 分层通信策略

在大规模多节点环境中，分层通信策略能够显著提高性能：

```python
from hcclang.optimization.distributors import create_hierarchical_algorithm

# 定义分层策略
hierarchical_algo = create_hierarchical_algorithm(
    global_topology=CM384_full(),
    local_topology_size=8,      # 每节点8个NPU
    intra_node_strategy="ring", # 节点内使用环形算法
    inter_node_strategy="tree", # 节点间使用树形算法
    root_selection="bandwidth_optimal"  # 带宽优化的根节点选择
)
```

### 性能优化技巧

为了获得最佳性能，建议遵循以下优化原则：

**带宽感知调度**：根据不同连接的带宽特性安排通信顺序，优先使用高带宽连接传输大数据块。

**延迟隐藏**：通过重叠计算和通信操作来隐藏网络延迟，特别是在多阶段算法中。

**缓存友好访问**：设计数据访问模式时考虑缓存局部性，减少内存访问开销。

**负载均衡**：确保所有节点的工作负载均衡，避免出现性能瓶颈。

### 调试和验证

HCCLang提供了完善的调试和验证工具：

```python
from hcclang.validation import verify_correctness, analyze_performance

# 验证算法正确性
is_correct = verify_correctness(algorithm, test_data)

# 性能分析
perf_report = analyze_performance(algorithm, topology)
print(f"预估通信时间：{perf_report.total_time_ms} ms")
print(f"带宽利用率：{perf_report.bandwidth_utilization:.2%}")
```

## 扩展和定制

HCCLang的架构设计充分考虑了扩展性，开发者可以轻松添加新的拓扑定义、优化策略和代码生成器。

### 自定义拓扑

创建新的拓扑定义需要继承基础拓扑类：

```python
from hcclang.topologies.base import Topology

class MyCustomTopology(Topology):
    def __init__(self, nodes, connections):
        super().__init__(nodes)
        self.setup_connections(connections)
  
    def setup_connections(self, connections):
        # 实现具体的连接逻辑
        pass
```

### 自定义优化器

添加新的优化策略：

```python
from hcclang.optimization.base import Optimizer

class MyOptimizer(Optimizer):
    def optimize(self, algorithm):
        # 实现优化逻辑
        return optimized_algorithm
```

## 项目状态和路线图

HCCLang项目目前已经具备了完整的核心功能，但仍在持续发展和完善中。

### 当前功能状态

**已完成功能**：

- ✅ 完整的DSL语法和编译器
- ✅ HCCL JSON代码生成
- ✅ CM384拓扑支持
- ✅ 标准算法库
- ✅ 交互式教程和文档
- ✅ 基础性能分析工具

**开发中功能**：

- 🚧 高级优化算法
- 🚧 自动化算法生成
- 🚧 运行时性能分析
- 🚧 可视化工具增强

**计划功能**：

- 📋 更多硬件平台支持
- 📋 算法正确性形式化验证
- 📋 机器学习辅助优化
- 📋 云原生部署支持

### 贡献指南

HCCLang欢迎社区贡献，无论是bug修复、功能增强还是文档改进。项目遵循开放协作的原则，鼓励研究人员和工程师共同推进集合通信算法的发展。

对于希望贡献代码的开发者，建议首先阅读项目的代码规范和设计文档，然后从小的功能改进开始，逐步熟悉系统架构。对于算法研究人员，可以通过提供新的算法实现和性能基准测试来为项目做出贡献。

## 许可证和致谢

HCCLang采用MIT许可证，基于Microsoft MSCCLang项目进行开发。我们感谢Microsoft Research团队为集合通信算法研究领域做出的重要贡献，以及华为昇腾团队在硬件平台支持方面提供的技术指导。

该项目的成功离不开开源社区的支持，特别是在算法验证、性能优化和文档编写方面。我们期待更多的研究人员和工程师加入到这个项目中来，共同推进高性能计算和人工智能领域的发展。
