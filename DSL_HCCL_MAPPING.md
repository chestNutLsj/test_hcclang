# DSL to HCCL API Mapping Specification

## 1. Overview

This document specifies the mapping from the high-level HCCLang Intermediate Representation (IR) to the concrete C++ code and API calls of the Huawei Collective Communication Library (HCCL). It serves as the core reference for the transpiler development.

The goal is to translate the abstract, platform-agnostic DSL operations into a set of compilable, performant C++ files that correctly implement the desired collective algorithm using the HCCL library.

## 2. Core Concept Mapping

| HCCLang IR Concept | HCCL C++ Equivalent                 | Notes                                                                                  |
| :----------------- | :---------------------------------- | :------------------------------------------------------------------------------------- |
| `Program`        | A set of generated C++ files        | The transpiler will output `*executor.h`, `*executor.cc`, `*alg.h`, `*alg.cc`. |
| `Gpu`            | A single rank                       | Represents one process in the collective operation.                                    |
| `Threadblock`    | A sequence of tasks on a `Stream` | A logical grouping of operations. All ops within a TB are executed sequentially.       |
| `Channel`        | A `hccl::Transport` object        | Represents a communication link between two specific ranks.                            |
| `Op`             | A specific HCCL API call            | The fundamental unit of translation. See the table below.                              |

---

## 3. Instruction (`Op`) Mapping

This is the primary mapping table for the transpiler.

| DSL `Op.inst`            | HCCL C++ Equivalent                   | Synchronization & Context                                                                                                                                                                                                                                             | Generated Code Example                                                                                                                                                                                                                                                                                                          |
| :------------------------- | :------------------------------------ | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `send`                   | `link->TxAsync(...)`                | **Before:** `link->TxAck(stream)` and `link->RxAck(stream)` for handshake. `<br>` **After:** `link->TxWaitDone(stream)` for completion.                                                                                                           | `CHK_RET(linkRight_->TxAck(stream_));<br>``CHK_RET(linkLeft_->RxAck(stream_));<br>``CHK_RET(linkRight_->TxAsync(UserMemType::OUTPUT_MEM, offset, src, len, stream_));<br>``CHK_RET(linkRight_->TxWaitDone(stream_));`                                                                                                   |
| `recv`                   | `link->RxAsync(...)`                | Handled by the corresponding `send` op's synchronization.                                                                                                                                                                                                           | `CHK_RET(linkLeft_->RxAsync(UserMemType::INPUT_MEM, offset, dst, len, stream_));<br>``CHK_RET(linkLeft_->RxWaitDone(stream_));`                                                                                                                                                                                             |
| `copy`                   | `HcclD2DMemcpyAsync(...)`           | Executed on a single stream. Dependencies are managed by the DAG.                                                                                                                                                                                                     | `CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, stream_));`                                                                                                                                                                                                                                                                |
| `reduce`                 | `RxAsync` + Local Compute           | First, receive data from a peer using `RxAsync`. The actual reduction is a local operation on the GPU, typically outside the scope of HCCL's direct APIs but can be represented as a placeholder for a kernel launch. For now, we map it to the data movement part. | `// Receive data for reduction<br>``CHK_RET(linkLeft_->RxAsync(UserMemType::INPUT_MEM, offset, dst, len, stream_));<br>``// Placeholder for local reduction kernel`                                                                                                                                                       |
| `rrc` (recv_reduce_copy) | `RxWithReduce` or `RxAsync`+Local | Can map to `RxWithReduce` if the hardware supports it. Otherwise, same as `reduce`.                                                                                                                                                                               | `// Option 1: Use dedicated API<br>``CHK_RET(linkLeft_->RxWithReduce(UserMemType::INPUT_MEM, offset, dst, len, reduceSrc, reduceDst, count, dataType, reduceOp, stream_, attr));<br>``// Option 2: Decompose<br>``CHK_RET(linkLeft_->RxAsync(UserMemType::INPUT_MEM, offset, dst, len, stream_)); /* + local reduce */` |
| `rrs` (recv_reduce_send) | `RxWithReduce` + `TxAsync`        | A combination of the `reduce` and `send` mappings.                                                                                                                                                                                                                | `// ... RxWithReduce or RxAsync ...<br>``// ... TxAsync ...`                                                                                                                                                                                                                                                                |
| `send_sync`              | `link->TxData(...)`                 | Synchronous send with built-in handshake and completion.                                                                                                                                                                                                              | `CHK_RET(linkRight_->TxData(UserMemType::OUTPUT_MEM, offset, src, len, stream_));`                                                                                                                                                                                                                                            |
| `recv_sync`              | `link->RxData(...)`                 | Synchronous receive with built-in handshake and completion.                                                                                                                                                                                                           | `CHK_RET(linkLeft_->RxData(UserMemType::INPUT_MEM, offset, dst, len, stream_));`                                                                                                                                                                                                                                              |
| `barrier`                | `ExecuteBarrier(...)`               | Collective barrier across specified links.                                                                                                                                                                                                                            | `CHK_RET(ExecuteBarrier(linkLeft_, linkRight_));`                                                                                                                                                                                                                                                                             |
| `prepare_tx`             | `link->TxPrepare(...)`              | Prepares the transmission channel.                                                                                                                                                                                                                                    | `CHK_RET(linkRight_->TxPrepare(stream_));`                                                                                                                                                                                                                                                                                    |
| `prepare_rx`             | `link->RxPrepare(...)`              | Prepares the reception channel.                                                                                                                                                                                                                                       | `CHK_RET(linkLeft_->RxPrepare(stream_));`                                                                                                                                                                                                                                                                                     |
| `tx_done`                | `link->TxDone(...)`                 | Signals transmission done.                                                                                                                                                                                                                                            | `CHK_RET(linkRight_->TxDone(stream_));`                                                                                                                                                                                                                                                                                       |
| `rx_done`                | `link->RxDone(...)`                 | Signals reception done.                                                                                                                                                                                                                                               | `CHK_RET(linkLeft_->RxDone(stream_));`                                                                                                                                                                                                                                                                                        |

---

## 4. Buffer and Chunk Mapping

The DSL's memory model maps cleanly to HCCL's `DeviceMem`.

| DSL Concept                       | HCCL C++ Equivalent               | Notes                                                                                                                                                                                                                                             |
| :-------------------------------- | :-------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `Buffer` (input/output/scratch) | `hccl::DeviceMem`               | The transpiler will manage three main `DeviceMem` objects corresponding to the DSL buffers.                                                                                                                                                     |
| `ChunkRef(buffer, index, size)` | `deviceMem.range(offset, size)` | A `ChunkRef` is a pointer to a slice of a buffer. This is perfectly represented by creating a new temporary `DeviceMem` object from a larger one using the `.range()` method. The `offset` is calculated from the `index` and `size`. |

---

## 5. Structural Mapping

The transpiler will generate a class structure that mirrors the existing HCCL algorithm implementations.

1. **Executor Files (`coll_{alg_name}_{topo_name}_executor.h/.cc`)**

   * This class is mostly boilerplate code.
   * It inherits from the appropriate base executor (e.g., `CollAllGatherExecutor`).
   * It uses the `REGISTER_EXECUTOR` macro to make itself visible to the framework.
   * Its primary role is to get the corresponding `AlgTemplate` and execute it.
2. **Algorithm Files (`{alg_name}_{topo_name}.h/.cc`)**

   * This is where the core logic resides.
   * The class inherits from `AlgTemplateBase`.
   * A `RunAsync` or similar method will be the entry point.
   * A core `Run{CollectiveName}` method will contain the main loop.
   * **This loop is the target of our translation**: The sequence of `Op`s from the DSL's `RankDAG` will be translated into a sequence of C++ statements (API calls, synchronization) inside this loop.

---

## 6. Example Walkthrough: Ring AllGather (1 step)

Let's trace a single step of the Ring AllGather algorithm.

**Hypothetical DSL `Op` Sequence for rank `r`:**

1. `Op(send, rank=r, src=ChunkRef(r, output, chunk_to_send), dst=ChunkRef(r+1, ...))`
2. `Op(recv, rank=r, src=ChunkRef(r-1, ...), dst=ChunkRef(r, output, chunk_to_recv))`

**Generated C++ in `AllGatherRing::RunAllGather`:**

```cpp
// Corresponds to the dependency graph: must sync before communication
// This is the handshake part.
CHK_RET(linkLeft_->TxAck(stream_));  // Notifies prev rank we are ready to TX
CHK_RET(linkRight_->RxAck(stream_)); // Waits for next rank to be ready to RX

// Corresponds to DSL Op 1: send
// linkRight_ is the transport channel to rank r+1
DeviceMem txChunk = outputMem_.range(chunkOffset, chunkSize);
CHK_RET(linkRight_->TxAsync(UserMemType::OUTPUT_MEM, chunkOffset, txChunk.ptr(), chunkSize, stream_));

// Corresponds to DSL Op 2: recv
// linkLeft_ is the transport channel from rank r-1
DeviceMem rxChunk = outputMem_.range(recvOffset, chunkSize);
CHK_RET(linkLeft_->RxAsync(UserMemType::INPUT_MEM, recvOffset, rxChunk.ptr(), chunkSize, stream_));

// Corresponds to the dependency graph: must wait for communication to complete
// This is the completion part.
CHK_RET(linkLeft_->RxWaitDone(stream_));
CHK_RET(linkRight_->TxWaitDone(stream_));
```

---

## 7. Future DSL Enhancements (HCCL Feature Gaps)

This section identifies key HCCL features that are not currently expressible in the DSL. This serves as a reference for future language development to enhance performance tuning and expressiveness.

### 7.1. Explicit Synchronization and Events

* **HCCL Capability**: Provides a rich set of synchronization primitives like `LocalNotify` (`Post`/`Wait`), `TxAck`/`RxAck`, and `TxWaitDone`/`RxWaitDone`. This allows for fine-grained control over execution flow.
* **DSL Gap**: The DSL currently implies dependencies through the DAG structure but lacks syntax to specify the *type* of synchronization.
* **Potential DSL Enhancement**: Introduce explicit synchronization instructions.
  * `barrier()`: A collective barrier across a set of ranks.
  * `event e = post()`: Create and post an event.
  * `wait_on(e)`: Wait for a specific event. This would allow for more complex dependency patterns than a simple DAG edge.

### 7.2. Memory Management and Types

* **HCCL Capability**: Differentiates between `DeviceMem` and `HostMem`, providing mechanisms to manage both.
* **DSL Gap**: The DSL abstracts memory into `input`, `output`, and `scratch` buffers without specifying their location (Host or Device).
* **Potential DSL Enhancement**: Add memory location specifiers to buffer definitions.
  * `buffer B(type=host, size=...)`: Define a buffer explicitly in host memory.
  * This would enable algorithms that require host-device interaction.

### 7.3. Stream Management

* **HCCL Capability**: Allows creation and management of multiple streams, including distinguishing between main and subordinate streams for concurrent execution.
* **DSL Gap**: The DSL assumes a single, implicit stream for all operations. It cannot express concurrent communication patterns on different streams.
* **Potential DSL Enhancement**: Introduce syntax for stream management.
  * `stream s1, s2`: Declare multiple streams.
  * `with stream(s1): ...`: A block of operations to be executed on a specific stream, enabling expression of overlapping communication and computation.

### 7.4. Low-Level Network Configuration

* **HCCL Capability**: Exposes APIs like `HcclNetInit`, `SetQosCfg`, and `AddWhiteList` for expert-level network tuning.
* **DSL Gap**: These details are completely abstracted away, which is generally desirable for a high-level DSL.
* **Potential DSL Enhancement**: For expert users, provide an optional configuration block.
  * `configure network { qos = high; }`: Allow setting specific network parameters for an algorithm. This should be an advanced feature and not part of the core language syntax.

---

## 8. Synchronization Mapping: From Implicit to Explicit

This section details how the DSL's implicit dependency model (`step` order and `depends` list) is mapped to explicit HCCL synchronization APIs.

### 8.1. Sequential Execution (`step` order)

* **DSL Semantics**: Operations within the same `Threadblock` are executed sequentially according to their `step` number.
* **HCCL Mapping**: This maps directly to the in-order execution property of an `hccl::Stream`. All C++ calls generated for a single DSL `Threadblock` will be enqueued on the same `hccl::Stream`, which guarantees their sequential execution. This is a direct 1-to-1 mapping.

### 8.2. Explicit Dependencies (`depends` list)

* **DSL Semantics**: The `op.depends` list creates an explicit dependency between two operations, often across different ranks or threadblocks.
* **HCCL Mapping**: This is mapped to explicit synchronization API calls, depending on the nature of the dependency.

| Dependency Type                        | DSL Representation                                      | HCCL Mapping & Protocol                                                                                                                                                                                                                                                                                                                                                        | Purpose                                                                                                                                                                                                                                                                                                     |
| :------------------------------------- | :------------------------------------------------------ | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Inter-Rank (Send/Recv)**       | A `recv` op `depends` on a `send` op from a peer. | **1. Readiness Handshake:**`<br>link->TxAck(stream)``<br>link->RxAck(stream)``<br>`**2. Data Transfer:**`<br>link->TxAsync(UserMemType::OUTPUT_MEM, offset, src, len, stream)``<br>link->RxAsync(UserMemType::INPUT_MEM, offset, dst, len, stream)``<br>`**3. Completion Sync:**`<br>link->TxWaitDone(stream)``<br>link->RxWaitDone(stream)` | The handshake ensures the receiver is ready before the sender transmits. The completion sync ensures the data transfer is finished before the next dependent operation begins.                                                                                                                              |
| **Intra-Rank (Inter-TB/Stream)** | An op in `TB1` `depends` on an op in `TB0`.       | `LocalNotify` object with `Post`/`Wait` calls.                                                                                                                                                                                                                                                                                                                           | One stream (`TB0`) calls `notify->Post(stream, dispatcher, stage)` after its operation. The dependent stream (`TB1`) calls `notify->Wait(stream, dispatcher, stage, timeout)` before its operation. This is crucial for managing dependencies between different concurrent streams on the same GPU. |
| **Ready/Finish Protocol**        | Cross-rank synchronization checkpoints.                 | **Ready Protocol:**`<br>link->PostReady(stream)``<br>link->WaitReady(stream)``<br>`**Finish Protocol:**`<br>link->PostFin(stream)``<br>link->WaitFin(stream)``<br>`**Finish Ack:**`<br>link->PostFinAck(stream)``<br>link->WaitFinAck(stream)`                                                                                               | Provides structured synchronization points for complex multi-stage algorithms. Ready signals indicate preparation completion, Finish signals indicate stage completion.                                                                                                                                     |
| **Data Reception Ack**           | Acknowledgment of data reception.                       | `link->DataReceivedAck(stream)`                                                                                                                                                                                                                                                                                                                                              | Explicit acknowledgment that data has been received and can be safely processed.                                                                                                                                                                                                                            |
