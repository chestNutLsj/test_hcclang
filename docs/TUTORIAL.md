# HCCLang Complete Tutorial: From DSL to HCCL C++ Code

## 概述

本教程展示了如何使用HCCLang DSL完整工作流，从上层算法描述到底层HCCL C++代码生成。我们以single ring allgather算法为例，演示整个流程。

## 目录

1. [环境准备](#环境准备)
2. [DSL算法编写](#dsl算法编写)
3. [序列化到JSON](#序列化到json)
4. [转换为MSCCL XML](#转换为msccl-xml)
5. [生成HCCL C++代码](#生成hccl-c代码)
6. [代码验证](#代码验证)
7. [运行示例](#运行示例)
8. [故障排除](#故障排除)

## 环境准备

### 1. 目录结构
```
hccl-tools/hcclang-demo/
├── hcclang/                    # HCCLang DSL核心模块
│   ├── core/                   # 算法和集合操作定义
│   ├── language/               # DSL语言构造
│   ├── topologies/            # 拓扑定义
│   ├── runtime/               # 代码生成运行时
│   └── solver/                # 求解器组件
├── docs/                      # 文档和示例
│   ├── tutorial_single_ring_allgather.py  # 完整教程脚本
│   ├── single-ring-allgather-template/    # HCCL模板文件
│   └── single-ring-allgather-impl/        # HCCL参考实现
└── tests/                     # 测试文件
```

### 2. 依赖库
- Python 3.8+
- lxml (XML处理)
- dataclasses (数据结构)

## DSL算法编写

### 1. 算法概念

Single Ring AllGather算法是集合通信中的基础操作：
- **输入**: 每个rank有一个数据块
- **输出**: 每个rank拥有所有rank的数据块
- **拓扑**: 4个rank形成环形连接 (0→1→2→3→0)

### 2. DSL代码实现

```python
def create_single_ring_allgather():
    """使用HCCLang DSL创建single ring allgather算法"""
    
    # 1. 定义拓扑：4节点环形
    topology = ring(4)  # 使用内置ring拓扑生成器
    
    # 2. 定义数据块和条件
    chunks = []
    for chunk_id in range(4):
        # 初始条件：chunk i 只在rank i上
        precondition = {chunk_id}
        # 最终条件：所有chunk在所有rank上
        postcondition = set(range(4))
        chunks.append(Chunk(precondition, postcondition, address=chunk_id))
    
    # 3. 定义集合操作
    collective = Collective(name='allgather', 
                           num_nodes=4, 
                           chunks=chunks,
                           runtime_name='AllGather')
    
    # 4. 定义输入输出映射
    input_map = {rank: {rank} for rank in range(4)}      # 每个rank只有自己的chunk
    output_map = {rank: set(range(4)) for rank in range(4)}  # 每个rank有所有chunk
    
    # 5. 生成算法步骤 (3步完成4-rank ring allgather)
    steps = []
    for step in range(3):
        sends = []
        for rank in range(4):
            src_rank = rank
            dst_rank = (rank + 1) % 4
            chunk_to_send = (rank - step) % 4
            sends.append([chunk_to_send, src_rank, dst_rank])
        steps.append(Step(rounds=1, sends=sends))
    
    # 6. 创建算法实例
    instance = Instance(steps=len(steps), chunks=4)
    algorithm = Algorithm(
        name='single_ring_allgather_4rank',
        collective=collective,
        topology=topology,
        instance=instance,
        steps=steps,
        input_map=input_map,
        output_map=output_map
    )
    
    return algorithm
```

### 3. 算法步骤详解

Ring AllGather的3个步骤：

**Step 0**: 初始数据分发
```
Rank 0: [0] → sends 0 to Rank 1
Rank 1: [1] → sends 1 to Rank 2  
Rank 2: [2] → sends 2 to Rank 3
Rank 3: [3] → sends 3 to Rank 0
```

**Step 1**: 第二轮传递
```
Rank 0: [0,3] → sends 3 to Rank 1
Rank 1: [1,0] → sends 0 to Rank 2
Rank 2: [2,1] → sends 1 to Rank 3
Rank 3: [3,2] → sends 2 to Rank 0
```

**Step 2**: 最终传递
```
Rank 0: [0,3,2] → sends 2 to Rank 1
Rank 1: [1,0,3] → sends 3 to Rank 2
Rank 2: [2,1,0] → sends 0 to Rank 3
Rank 3: [3,2,1] → sends 1 to Rank 0
```

**结果**: 每个rank都有 [0,1,2,3]

## 序列化到JSON

### 1. JSON格式说明

HCCLang使用JSON作为中间表示格式：

```json
{
    "msccl_type": "algorithm",
    "name": "single_ring_allgather_4rank",
    "instance": {
        "msccl_type": "instance",
        "steps": 3,
        "chunks": 4,
        "pipeline": null
    },
    "input_map": {"0": [0], "1": [1], "2": [2], "3": [3]},
    "output_map": {"0": [0,1,2,3], "1": [0,1,2,3], "2": [0,1,2,3], "3": [0,1,2,3]},
    "steps": [
        {
            "msccl_type": "step", 
            "rounds": 1,
            "sends": [[0,0,1], [1,1,2], [2,2,3], [3,3,0]]
        }
        // ... 更多步骤
    ],
    "collective": { /* 集合操作定义 */ },
    "topology": { /* 拓扑定义 */ }
}
```

### 2. 序列化代码

```python
# 保存算法到JSON文件
save_msccl_object(algorithm, 'single_ring_allgather.json')

# 加载算法从JSON文件
loaded_algorithm = load_msccl_object('single_ring_allgather.json')
```

## 转换为MSCCL XML

### 1. XML格式说明

MSCCL XML是执行引擎使用的格式，包含详细的操作序列：

```xml
<algo name="single_ring_allgather_4rank" proto="Simple" nchannels="1" ngpus="4" coll="AllGather">
  <gpu id="0" i_chunks="1" o_chunks="4" s_chunks="0">
    <tb id="0" send="-1" recv="3" chan="0">
      <step s="0" type="r" srcbuf="i" srcoff="0" dstbuf="o" dstoff="3" cnt="1" depid="-1" deps="-1" hasdep="1"/>
      <step s="1" type="r" srcbuf="o" srcoff="2" dstbuf="o" dstoff="2" cnt="1" depid="-1" deps="-1" hasdep="1"/>
    </tb>
    <tb id="1" send="1" recv="-1" chan="0">
      <step s="0" type="s" srcbuf="i" srcoff="0" dstbuf="o" dstoff="0" cnt="1" depid="-1" deps="-1" hasdep="0"/>
      <step s="1" type="s" srcbuf="o" srcoff="3" dstbuf="o" dstoff="3" cnt="1" depid="0" deps="0" hasdep="0"/>
    </tb>
  </gpu>
  <!-- 更多GPU定义 -->
</algo>
```

### 2. XML元素说明

- **`<algo>`**: 算法根元素，包含元数据
- **`<gpu>`**: 每个GPU/rank的定义
- **`<tb>`**: 线程块，处理特定通信模式
- **`<step>`**: 具体的操作步骤

### 3. 转换代码

```python
# 使用ncclize转换算法为XML
xml_content = ncclize(algorithm, pretty_print=True)

# 保存XML到文件
with open('single_ring_allgather.xml', 'w') as f:
    f.write(xml_content)
```

## 生成HCCL C++代码

### 1. 增强的XML解析器

我们开发了专门的XML解析器来处理MSCCL XML：

```python
class XmlAlgorithmParser:
    """解析MSCCL XML文件的专用解析器"""
    
    def __init__(self, xml_file_path: str):
        self.tree = ET.parse(xml_file_path)
        self.root = self.tree.getroot()
        
        # 解析算法元数据
        self.algo_name = self.root.get('name')
        self.collective_type = self.root.get('coll')
        self.num_gpus = int(self.root.get('ngpus'))
        
        # 解析GPU和操作详情
        self.gpus = self._parse_gpus()
        self.operations = self._extract_all_operations()
```

### 2. HCCL代码生成器

```python
class HcclXmlExecutorGenerator:
    """增强的HCCL Executor代码生成器，解析XML输入"""
    
    def generate_executor_files(self, output_dir: str) -> Tuple[str, str]:
        """生成.h和.cc文件"""
        header_content = self._generate_header_file()
        source_content = self._generate_source_file()
        
        # 保存文件
        base_name = f"{self.algo_name.lower()}_executor"
        header_path = os.path.join(output_dir, f"{base_name}.h")
        source_path = os.path.join(output_dir, f"{base_name}.cc")
        
        return header_path, source_path
```

### 3. 生成的代码特点

#### 头文件结构：
```cpp
class single_ring_allgather_4rankExecutor : public CollAllGatherExecutor {
public:
    explicit single_ring_allgather_4rankExecutor(const HcclDispatcher dispatcher, 
                                                 std::unique_ptr<TopoMatcher>& topoMatcher);
    
private:
    // 资源计算
    HcclResult CalcStreamNum(u32& streamNum) override;
    HcclResult CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport) override;
    
    // 算法编排  
    HcclResult KernelRun(const OpParam &param, ExecMem &execMem) override;
    
    // XML驱动的执行
    HcclResult ExecuteStep(u32 stepId, const OpParam& param, ExecMem& execMem);
    HcclResult ExecuteSend(u32 srcRank, u32 dstRank, DeviceMem srcMem, u64 offset, u64 size, const OpParam& param);
    HcclResult ExecuteReceive(u32 srcRank, u32 dstRank, DeviceMem dstMem, u64 offset, u64 size, const OpParam& param);
    
    static constexpr u32 NUM_RANKS = 4;
    static constexpr u32 TOTAL_STEPS = 3;
};
```

#### 核心执行逻辑：
```cpp
HcclResult KernelRun(const OpParam &param, ExecMem &execMem) {
    // 基于XML规范执行算法的所有步骤
    for (u32 step = 0; step < TOTAL_STEPS; step++) {
        CHK_RET(ExecuteStep(step, param, execMem));
    }
    return HCCL_SUCCESS;
}

HcclResult ExecuteStep(u32 stepId, const OpParam& param, ExecMem& execMem) {
    switch (stepId) {
        case 0:
            // 执行Step 0的所有操作...
            if (param.rank == 0) {
                CHK_RET(ExecuteSend(0, 1, srcMem, offset, size, param));
            }
            break;
        // 更多步骤...
    }
    return HCCL_SUCCESS;
}
```

## 代码验证

### 1. 功能验证

我们通过以下方式验证生成代码的正确性：

1. **结构验证**: 检查类继承关系和接口实现
2. **逻辑验证**: 对比算法步骤与预期行为
3. **API验证**: 确认HCCL API调用的正确性

### 2. 与参考实现对比

| 功能特性 | 生成代码 | 参考实现 | 验证结果 |
|---------|---------|---------|----------|
| 基本框架 | ✅ 正确 | ✅ 完整 | 通过 |
| 资源计算 | ✅ 基本功能 | ✅ 完整功能 | 部分通过 |
| 单环AllGather | ✅ 实现 | ✅ 实现 | 通过 |
| 多环支持 | ❌ 缺失 | ✅ 支持 | 未通过 |
| 错误处理 | 🟡 基本 | ✅ 完整 | 部分通过 |
| 性能优化 | ❌ 缺失 | ✅ 优化 | 未通过 |

### 3. 详细分析报告

详细的验证结果请参考 [代码验证报告](code_validation_report.md)。

## 运行示例

### 1. 完整流程执行

```bash
# 进入项目目录
cd hccl-tools/hcclang-demo

# 运行完整教程
python docs/tutorial_single_ring_allgather.py
```

### 2. 预期输出

```
=== HCCLang Single Ring AllGather Tutorial ===

1. Creating algorithm using HCCLang DSL...
   Algorithm: single_ring_allgather_4rank
   Topology: Ring(n=4)
   Ranks: 4
   Steps: 3

2. Serializing algorithm to JSON...
   Saved to: docs/single_ring_allgather.json

3. Converting JSON to MSCCL XML...
   Generated XML: docs/single_ring_allgather.xml

4. Converting XML to HCCL C++ Executor...
   Generated XML-based header: docs/single_ring_allgather_4rank_executor.h
   Generated XML-based source: docs/single_ring_allgather_4rank_executor.cc
   Algorithm type detected: AllGather
   Total communication steps: 3

=== Tutorial Complete ===
```

### 3. 生成的文件

- `single_ring_allgather.json`: 算法的JSON表示
- `single_ring_allgather.xml`: MSCCL XML格式
- `single_ring_allgather_4rank_executor.h`: HCCL头文件
- `single_ring_allgather_4rank_executor.cc`: HCCL源文件

## 故障排除

### 1. 常见错误

#### 导入错误
```python
ModuleNotFoundError: No module named 'hcclang'
```
**解决方案**: 确保Python路径正确设置
```python
sys.path.insert(0, hcclang_demo_dir)
```

#### XML解析错误
```
xml.etree.ElementTree.ParseError: not well-formed
```
**解决方案**: 检查XML文件格式，确保ncclize正常运行

#### 代码生成错误
```
AttributeError: 'NoneType' object has no attribute 'pipeline'
```
**解决方案**: 确保创建了正确的Instance对象
```python
instance = Instance(steps=len(steps), chunks=4)
```

### 2. 调试技巧

1. **启用详细日志**:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **检查中间文件**:
   - 查看生成的JSON文件是否正确
   - 验证XML文件的格式和内容
   - 检查C++代码的语法

3. **使用参考实现对比**:
   - 对比算法步骤
   - 检查API调用差异
   - 验证数据流向

## 扩展和定制

### 1. 支持更多rank数

修改DSL算法创建函数：
```python
def create_ring_allgather(num_ranks):
    topology = ring(num_ranks)
    # 适配num_ranks的逻辑...
```

### 2. 支持其他集合操作

```python
def create_allreduce_algorithm():
    collective = Collective(name='allreduce', ...)
    # 实现AllReduce逻辑...
```

### 3. 自定义拓扑

```python
def create_custom_topology():
    links = [[0,1,0,1], [1,0,1,0], ...]  # 自定义连接矩阵
    topology = Topology('custom', links)
```

## 总结

本教程展示了HCCLang的完整工作流程：

1. **DSL编程**: 使用高级DSL描述算法逻辑
2. **自动转换**: 通过工具链自动生成底层代码
3. **代码验证**: 确保生成代码的正确性
4. **性能优化**: 识别并改进性能瓶颈

HCCLang为集合通信算法开发提供了强大的抽象层，大大简化了从算法设计到实现的过程。虽然当前实现在某些企业级特性上还有改进空间，但已经为教学、研究和原型开发提供了良好的基础。

### 后续工作

1. **功能增强**: 实现多环支持、错误恢复等企业级特性
2. **性能优化**: 添加自动调优和性能分析功能  
3. **工具完善**: 开发可视化调试和性能分析工具
4. **生态建设**: 扩展支持更多硬件平台和通信库