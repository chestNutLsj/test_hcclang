# HCCLang - 华为HCCL集合通信算法领域特定语言

HCCLang是一个专为华为集合通信库(HCCL)设计的领域特定语言(DSL)，用于定义和生成高效的集合通信算法。该项目基于Microsoft的MSCCLang进行适配开发，为HCCL环境提供了完整的算法描述、优化和代码生成解决方案。

## 项目概述

HCCLang提供了一套完整的工具链，让开发者能够以高级、直观的方式描述集合通信算法，并自动生成针对HCCL可编译代码。该项目的核心价值在于弥合算法设计与硬件实现之间的鸿沟，使研究人员和工程师能够专注于算法逻辑本身，而无需深入底层实现细节。

HCCLang的设计遵循分离关注点的原则，将算法描述、优化策略和代码生成解耦。算法设计者可以使用高级抽象描述通信模式，优化专家可以应用各种变换和组合策略，而系统工程师则可以专注于运行时集成和性能调优。这种分层设计不仅提高了开发效率，也增强了系统的可维护性和扩展性。

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
```

## 快速入门指南

HCCLang的学习曲线被设计得相对平缓，开发者可以从简单的示例开始，逐步掌握高级特性。

### 基础算法示例

以下是一个完整的Mesh AllGather算法实现示例，展示了HCCLang的基本使用模式：

```python
import os
import sys

# Add hcclang to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from hcclang.language import HCCLProgram, chunk, Check, Buffer
from hcclang.language.collectives import AllGather
from hcclang.topologies.generic import fully_connected
from hcclang.runtime.hcclize import DSLToHcclTranspiler, HcclCodeGenConfig, CollectiveType, TopologyType

def mesh_allgather_algorithm(num_ranks=4):
    """
    Implement mesh allgather algorithm using HCCLang DSL.
  
    Args:
        num_ranks: Number of ranks in the mesh topology
  
    Returns:
        HCCLProgram instance with mesh allgather implementation
    """
    print(f"Creating mesh allgather algorithm for {num_ranks} ranks")
  
    # Create fully connected topology as mesh
    topology = fully_connected(num_ranks)
    print(f"✓ Created mesh topology: {topology.name}")
  
    # Create AllGather collective (non-inplace)
    # chunk_factor=1 means each rank starts with 1 chunk
    collective = AllGather(num_ranks=num_ranks, chunk_factor=1, inplace=False)
    print(f"✓ Created AllGather collective: {collective.name}")
  
    # Create HCCLProgram with mesh allgather implementation
    with HCCLProgram(
        name=f"mesh_allgather_{num_ranks}ranks",
        topo=topology,
        collective=collective,
        instances=1,
        protocol='Simple'
    ) as prog:
        print(f"✓ Created HCCLProgram: {prog.name}")
        print(f"  - Ranks: {prog.num_ranks}")
        print(f"  - Protocol: {prog.protocol}")
  
        # Implement mesh allgather algorithm
        # In mesh (fully connected) allgather, all ranks can communicate simultaneously
        # Each rank receives data from all other ranks in parallel
  
        # Step 1: Each rank copies its own data to output buffer
        for rank in range(num_ranks):
            own_chunk = chunk(rank, Buffer.input, 0, 1)  # Own chunk from input buffer
            own_chunk.copy(rank, Buffer.output, rank)    # Copy to output buffer at position rank
            print(f"  Rank {rank}: copied own chunk to output buffer position {rank}")
  
        # Step 2: All-to-all data exchange in mesh topology
        # In mesh topology, each rank can communicate with all other ranks simultaneously
        # We'll implement a simplified mesh pattern where each rank receives data from all others
        for step in range(num_ranks - 1):
            print(f"\n--- Step {step} ---")
            for rank in range(num_ranks):
                # Each rank receives from one other rank per step in round-robin fashion
                src_rank = (rank + step + 1) % num_ranks
  
                # In mesh allgather, we need to simulate receiving data from src_rank
                # Create a receive operation from src_rank to current rank
                # The chunk being sent is from src_rank's original position
                src_chunk = chunk(src_rank, Buffer.output, src_rank, 1)  # Source data from src_rank
  
                # Copy the chunk to current rank's output buffer at the source's position
                dst_chunk = src_chunk.copy(rank, Buffer.output, src_rank)
                print(f"  Rank {rank} <- Rank {src_rank}: mesh communication")
                print(f"    ✓ Received chunk from rank {src_rank} at position {src_rank}")
  
        print(f"\n✓ Mesh AllGather algorithm implementation complete")
  
        return prog

def main():
    """Test mesh allgather with 8 ranks."""
    print("=== HCCLang Mesh AllGather Implementation ===")
    print()
  
    num_ranks = 8
    print("=" * 50)
    print(f"Testing Mesh AllGather with {num_ranks} ranks")
    print("=" * 50)
  
    # Create the algorithm
    program = mesh_allgather_algorithm(num_ranks)
  
    print()
    print("=== Generating Mesh HCCL C++ Code ===")
  
    # Configure code generation
    output_dir = f"generated_mesh_allgather_{num_ranks}ranks"
    template_dir = os.path.join(os.path.dirname(__file__), "..", "..", "hcclang", "runtime", "templates")
  
    config = HcclCodeGenConfig(
        collective=CollectiveType.ALLGATHER,
        topology=TopologyType.MESH,
        output_dir=output_dir,
        template_dir=template_dir,
        algorithm_name=program.name,
        num_ranks=program.num_ranks,
        num_steps=0  # Will be calculated from program
    )
  
    # Initialize transpiler
    transpiler = DSLToHcclTranspiler(config)
  
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
  
        # Generate C++ code files
        # First convert HCCLProgram to lower-level Program representation
        lower_program = program.lower()
  
        # Debug: Print analysis results from enhanced transpiler
        print(f"\n--- Transpiler Analysis Debug ---")
        analysis = transpiler._analyze_communication_pattern(lower_program)
        print(f"DSL Program Analysis Results:")
        print(f"  - Total steps: {analysis['total_steps']}")
        print(f"  - Max rank: {analysis['max_rank']}")
        print(f"  - Number of mesh connections: {len(analysis['communication_pairs'])}")
        print(f"  - Communication pairs: {list(analysis['communication_pairs'])[:10]}...")  # Show first 10 pairs
        print(f"  - Total communication pairs: {len(analysis['communication_pairs'])}")
        print(f"  - Communication phases: {analysis['communication_phases']}")
        for i, phase in enumerate(analysis['communication_phases'], 1):
            print(f"    Phase {i}: {phase}")
        print(f"  - Pattern: {analysis.get('pattern', 'NOT_SET')}")
        print(f"  - Communication pattern: {analysis.get('communication_pattern', 'NOT_SET')}")
        print(f"  - Topology type: {analysis.get('topology_type', 'NOT_SET')}")
        print(f"  - Peer calculation: {analysis.get('peer_calculation', 'NOT_SET')}")
  
        # Generate code using the transpiler
        generated_files = transpiler.transpile_program(lower_program)
  
        print(f"\n--- Generated Algorithm Steps Preview (first 800 chars) ---")
        if 'alg_source' in generated_files:
            try:
                with open(generated_files['alg_source'], 'r') as f:
                    content = f.read()
                    # Find the algorithm implementation
                    if "AllGather Algorithm Implementation" in content:
                        start = content.find("AllGather Algorithm Implementation")
                        preview = content[start:start+800]
                        print(preview)
                    else:
                        print(content[:800])
            except Exception as e:
                print(f"Could not read generated file: {e}")
  
        print(f"\nGenerated {len(generated_files)} C++ files:")
        for file_type, file_path in generated_files.items():
            print(f"  - {file_type}: {file_path}")
  
        print(f"\n--- RunAllGather Function Preview in alg_source ---")
        if 'alg_source' in generated_files:
            try:
                with open(generated_files['alg_source'], 'r') as f:
                    content = f.read()
                    # Find RunAllGather function
                    if "RunAllGather" in content:
                        start = content.find("HcclResult AllgatherMesh::RunAllGather")
                        if start == -1:
                            start = content.find("RunAllGather")
                        end = content.find("}", start)
                        if end != -1:
                            preview = content[start:end+1]
                            # Show last 200 chars
                            print(preview[-200:])
                        else:
                            print("Could not find end of RunAllGather function")
                    else:
                        print("RunAllGather function not found")
            except Exception as e:
                print(f"Could not read generated file: {e}")
  
        print(f"✅ Successfully generated HCCL code for {num_ranks} ranks")
        print(f"   Output directory: {os.path.dirname(generated_files.get('alg_source', ''))}")
  
        print(f"\n--- Verifying DSL-to-HCCL Mappings ---")
        for file_type, file_path in generated_files.items():
            if file_path.endswith('.cc'):
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                        operations = []
                        if "copy" in content.lower():
                            operations.append("copy operation")
                        if "send" in content.lower():
                            operations.append("send operation")
                        if "recv" in content.lower():
                            operations.append("recv operation")
                        if "txasync" in content.lower():
                            operations.append("txasync operation")
                        if "rxasync" in content.lower():
                            operations.append("rxasync operation")
  
                        unsupported = []
                        if "TODO" in content:
                            unsupported.append("TODO markers")
                        if "NOT_IMPLEMENTED" in content:
                            unsupported.append("NOT_IMPLEMENTED")
  
                        operations_str = ", ".join(operations) if operations else "no operations"
                        unsupported_str = ", ".join(unsupported) if unsupported else "no unsupported operations"
                        print(f"   File {os.path.basename(file_path)}: {operations_str}, {unsupported_str}")
                except Exception as e:
                    print(f"   File {os.path.basename(file_path)}: could not analyze - {e}")
  
    except Exception as e:
        print(f"❌ Code generation failed: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()
```

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
from hcclang.topologies.cm384 import cm384_full

# 标准拓扑
ring_topo = ring(8)           # 8节点环形
tree_topo = tree(16)          # 16节点树形
mesh_topo = mesh(4, 4)        # 4x4网格

# 专用拓扑
cm384_topo = cm384_full() # CM384 384-NPU配置

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

## 扩展和定制

HCCLang的架构设计充分考虑了扩展性，开发者可以轻松添加新的拓扑定义、优化策略和代码生成器。

### 自定义拓扑

创建新的拓扑定义需要继承基础拓扑类：

```python
from hcclang.topologies.topo_tools import Topology

class MyCustomTopology(Topology):
    def __init__(self, nodes, connections):
        super().__init__(nodes)
        self.setup_connections(connections)
  
    def setup_connections(self, connections):
        # 实现具体的连接逻辑
        pass
```

## 项目状态和路线图

HCCLang项目目前已经初步的转译功能，但仍在持续发展和完善中。

### 当前功能状态

**已完成功能**：

- ✅ 完整的DSL语法
- ✅ 支持 AllGather 和 AlltoAll 算子的部分转译（Ring/Mesh）

**开发中功能**：

- 🚧 更多的算子、算法支持
- 🚧 自动的优化器支持
- 🚧 对算法的验证器支持

## 许可证和致谢

HCCLang采用GPLv2许可证，基于Microsoft MSCCLang项目进行开发。我们感谢Microsoft Research团队为集合通信算法研究领域做出的重要贡献，以及华为昇腾团队在硬件平台支持方面提供的技术指导。

该项目的成功离不开开源社区的支持，特别是在算法验证、性能优化和文档编写方面。我们期待更多的研究人员和工程师加入到这个项目中来，共同推进高性能计算和人工智能领域的发展。
