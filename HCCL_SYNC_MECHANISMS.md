# HCCL同步机制详解文档

## 概述

在HCCL (Huawei Collective Communication Library) 中，有两种主要的同步机制用于不同场景的协调控制：

1. **LocalNotify::Post/Wait** - 流内等待/流间同步
2. **link->PostFinAck/WaitFinAck** - 网络通信同步

本文档基于CANN-HCCL源码分析，详细解释两种机制的区别、用法和最佳实践。

## 1. LocalNotify::Post/Wait - 流内等待/流间同步

### 1.1 API定义

```cpp
// 发送通知
static HcclResult Post(Stream& stream, HcclDispatcher dispatcherPtr, 
                      const std::shared_ptr<LocalNotify> &notify, 
                      s32 stage = INVALID_VALUE_STAGE)

// 等待通知  
static HcclResult Wait(Stream& stream, HcclDispatcher dispatcherPtr, 
                      const std::shared_ptr<LocalNotify> &notify, 
                      s32 stage = INVALID_VALUE_STAGE, 
                      u32 timeOut = NOTIFY_DEFAULT_WAIT_TIME)
```

### 1.2 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| stream | Stream& | 执行同步操作的流对象 |
| dispatcherPtr | HcclDispatcher | Dispatcher句柄 |
| notify | std::shared_ptr<LocalNotify> | 本地通知对象指针 |
| stage | s32 | 算法阶段标识（可选） |
| timeOut | u32 | 等待超时时间（仅Wait接口） |

### 1.3 功能特点

- **作用域**：单NPU内流间同步
- **延迟**：极低（纳秒级）
- **通信媒介**：本地内存/信号量
- **故障域**：单NPU故障

## 2. link->PostFinAck/WaitFinAck - 网络通信同步

### 2.1 API定义

```cpp
// 发送完成应答信号
HcclResult PostFinAck(Stream &stream)

// 等待完成应答信号
HcclResult WaitFinAck(Stream &stream)
```

### 2.2 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| stream | Stream& | 执行同步操作的流对象 |

### 2.3 功能特点

- **作用域**：NPU间网络通信同步
- **延迟**：较高（微秒级，取决于网络）
- **通信媒介**：物理网络链路（UB/RDMA）
- **故障域**：网络链路故障

## 3. 两种机制对比

| 维度 | LocalNotify::Post/Wait | link->PostFinAck/WaitFinAck |
|------|----------------------|----------------------------|
| **作用域** | **单NPU内**流间同步 | **NPU间**网络通信同步 |
| **同步对象** | 本地多个Stream | 远程NPU通信链路 |
| **通信媒介** | 本地内存/信号量 | 物理网络链路（UB/RDMA） |
| **延迟** | 极低（纳秒级） | 较高（微秒级，取决于网络） |
| **故障域** | 单NPU故障 | 网络链路故障 |
| **带宽占用** | 无 | 占用网络带宽 |
| **可靠性** | 本地可靠 | 端到端可靠 |

## 4. 详细用法说明

### 4.1 LocalNotify典型用法

#### 多环并行算法同步模式

```cpp
// 从流等待主流的启动通知
ret = LocalNotify::Wait(algResResp_->slaveStreams[ringIndex], dispatcher_,
                       algResResp_->notifiesAux[ringIndex], profStage);
CHK_PRT_RET(ret != HCCL_SUCCESS, 
    HCCL_ERROR("[CollCommExecutor]stream[%u] wait failed", ringIndex), ret);

// 执行算法逻辑
ret = tempAlg->Prepare(inputMem, outputMem, outputMem, count, dataType,
    algResResp_->slaveStreams[ringIndex], reductionOp, LEVEL0_BRIDGE_RANK_ID, 
    singleRingSliceZero, baseOffset, ringNics[ringIndex]);

ret = RunTemplate(tempAlg, level0RingCommInfo);
CHK_PRT_RET(ret != HCCL_SUCCESS,
    HCCL_ERROR("[CollCommExecutor]stream[%u] run failed", ringIndex), ret);

// 从流通知主流任务完成
ret = LocalNotify::Post(algResResp_->slaveStreams[ringIndex], dispatcher_, 
                       algResResp_->notifiesMain[ringIndex], profStage);
CHK_PRT_RET(ret != HCCL_SUCCESS,
    HCCL_ERROR("[CollCommExecutor]stream[%u] record failed", ringIndex), ret);

// 主流发送给从流的下一阶段通知
ret = LocalNotify::Post(stream, dispatcher_, algResResp_->notifiesAux[ringIndex], profStage);
CHK_PRT_RET(ret != HCCL_SUCCESS,
    HCCL_ERROR("[CollCommExecutor]stream[%u] record failed", ringIndex), ret);
```

#### 主流等待从流完成模式

```cpp
// 主流等待所有从流完成
for (u32 ring = 0; ring < (ringNum - 1); ring++) {
    ret = LocalNotify::Wait(stream, dispatcher_, algResResp_->notifiesMain[ring], profStage);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollCommExecutor]stream wait main[%u] failed", ring), ret);
}
```

### 4.2 link->PostFinAck/WaitFinAck典型用法

#### 点对点通信完成确认模式

```cpp
// 网络通信操作序列
for (size_t i = 0; i < links.size(); i++) {
    if (links[i] == nullptr) {
        continue;
    }
    
    // 1. 执行数据传输
    u64 size = std::min(execMem.inputMem.size(), HCCL_INPLACE_MEMCOPY_SIZE);
    CHK_RET(links[i]->TxAsync(UserMemType::INPUT_MEM, 0, 
                             execMem.inputMem.ptr(), size, param.stream));
    CHK_RET(links[i]->RxAsync(UserMemType::INPUT_MEM, 0, 
                             execMem.inputMem.ptr(), size, param.stream));
    
    // 2. 发送完成确认信号
    CHK_RET(links[i]->PostFinAck(param.stream));
    
    // 3. 等待对方完成确认
    CHK_RET(links[i]->WaitFinAck(param.stream));
}
```

#### 网络链路健康检测模式

```cpp
// 收发信号校验链路可用性
for (size_t i = 0; i < links.size(); i++) {
    if (links[i] == nullptr) {
        continue;
    }
    
    // 握手确认链路连通性
    CHK_RET(links[i]->TxAck(param.stream));
    CHK_RET(links[i]->RxAck(param.stream));
    CHK_RET(links[i]->TxDataSignal(param.stream));
    CHK_RET(links[i]->RxDataSignal(param.stream));
}
```

## 5. 适用场景

### 5.1 LocalNotify适用场景

1. **多环并行算法**：协调不同环（ring）之间的执行顺序
2. **流水线同步**：确保前后阶段的数据依赖关系
3. **资源竞争控制**：避免多流同时访问共享资源
4. **算法阶段切换**：在算法的不同phase之间进行同步
5. **主从流协调**：主流控制多个从流的执行时序

### 5.2 link->PostFinAck/WaitFinAck适用场景

1. **点对点通信同步**：确保send/recv操作完全完成
2. **网络故障检测**：通过握手确认链路可用性
3. **数据一致性保证**：避免数据竞争和乱序问题
4. **通信完成确认**：确保远程NPU已接收/发送完成
5. **网络拥塞控制**：通过同步机制控制发送速率

## 6. 最佳实践

### 6.1 选择原则

**使用LocalNotify::Post/Wait当：**
- 需要协调同一NPU上的多个Stream
- 实现算法的不同阶段同步
- 控制并发资源访问
- 延迟敏感的同步操作

**使用link->PostFinAck/WaitFinAck当：**
- 执行网络通信操作后
- 需要确认远程NPU操作完成
- 实现可靠的点对点同步
- 检测网络链路健康状态

### 6.2 性能考虑

#### LocalNotify优势：
- **延迟极低**：适合频繁同步操作
- **不占用网络带宽**：纯本地操作
- **故障恢复简单**：单NPU内故障域
- **扩展性好**：支持多对多同步

#### link同步优势：
- **端到端可靠性**：提供完整的通信保证
- **网络故障检测**：能够发现链路问题
- **与通信紧密耦合**：确保数据传输完整性
- **硬件加速**：利用网络硬件的同步能力

### 6.3 典型组合模式

```cpp
// 完整的多流协调 + 网络通信模式
void MultiStreamNetworkOperation() {
    // 1. 流间协调 - 等待启动信号
    LocalNotify::Wait(slaveStream, dispatcher, notify_aux);
    
    // 2. 网络通信 - 数据传输
    link->TxAsync(UserMemType::OUTPUT_MEM, offset, srcData, size, stream);
    link->RxAsync(UserMemType::INPUT_MEM, offset, dstData, size, stream);
    
    // 3. 网络同步 - 确认传输完成
    link->PostFinAck(stream);
    link->WaitFinAck(stream);
    
    // 4. 流间协调 - 通知主流完成
    LocalNotify::Post(slaveStream, dispatcher, notify_main);
}
```

### 6.4 错误处理建议

```cpp
// 带超时的等待模式
HcclResult WaitWithTimeout(Stream& stream, std::shared_ptr<LocalNotify> notify, 
                          u32 timeout = 5000) {  // 5秒超时
    HcclResult ret = LocalNotify::Wait(stream, dispatcher_, notify, 
                                      INVALID_VALUE_STAGE, timeout);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("LocalNotify wait timeout or failed, ret=%d", ret);
        // 执行清理逻辑
        return ret;
    }
    return HCCL_SUCCESS;
}

// 网络链路健康检查
HcclResult CheckLinkHealth(LINK link, Stream& stream) {
    HcclResult ret = link->PostFinAck(stream);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("Link PostFinAck failed, link may be broken");
        return ret;
    }
    
    ret = link->WaitFinAck(stream);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("Link WaitFinAck timeout, remote node may be down");
        return ret;
    }
    
    return HCCL_SUCCESS;
}
```

## 7. Stream数量与物理带宽的对应关系补充

基于CM384网络拓扑的分析：

### 7.1 CM384物理架构
- **每个NPU连接到7个UB Switch**，每个UB端口带宽为**56GB/s**
- **理论聚合带宽**：7 × 56GB/s = **392GB/s**（节点内）

### 7.2 Stream与带宽关系
- **单Stream限制**：单个Stream通常对应单个数据流，无法同时利用所有7个UB端口的带宽
- **多Stream并行**：要充分利用392GB/s聚合带宽，需要多个Stream并行工作
- **实际带宽**：由于硬件调度、内存带宽等因素，实际可达带宽会低于理论峰值

## 8. TxAsync/RxAsync参数详解补充

### 8.1 API对比

**TxAsync（发送）：**
```cpp
HcclResult TxAsync(UserMemType dstMemType, u64 dstOffset, const void *src, u64 len, Stream &stream)
```

**RxAsync（接收）：**  
```cpp
HcclResult RxAsync(UserMemType srcMemType, u64 srcOffset, void *dst, u64 len, Stream &stream)
```

### 8.2 参数含义详解

| 参数 | TxAsync含义 | RxAsync含义 | 关键区别 |
|------|------------|------------|----------|
| **MemType** | `dstMemType`：**对端**内存类型 | `srcMemType`：**对端**内存类型 | 都指向远端内存 |
| **offset** | `dstOffset`：**对端内存**偏移 | `srcOffset`：**对端内存**偏移 | 都是远端内存的offset |
| **地址参数** | `src`：**本地源**地址 | `dst`：**本地目标**地址 | 都指向本地内存 |

### 8.3 关键理解点

1. **Offset始终针对远端内存**：无论TxAsync还是RxAsync，offset参数都指定远端内存的偏移量
2. **本地地址直接指定**：src/dst参数直接指向本地内存地址，无需额外offset
3. **对称设计**：TxAsync的dstOffset对应RxAsync的srcOffset，都表示通信对端的内存偏移

## 9. 总结

HCCL提供的两种同步机制各有特点和适用场景：

- **LocalNotify::Post/Wait** 专注于**本地流间协调**，提供低延迟、高效率的同步能力
- **link->PostFinAck/WaitFinAck** 专注于**网络通信确认**，提供端到端的可靠性保证

在实际使用中，两者通常配合使用，构建完整的分布式集合通信算法同步机制。正确选择和使用这些同步原语，是实现高性能、高可靠性集合通信算法的关键。

同时，理解Stream与物理带宽的关系以及TxAsync/RxAsync的参数语义，对于优化通信性能和正确实现算法逻辑至关重要。