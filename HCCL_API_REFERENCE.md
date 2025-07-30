# HCCL Custom Algorithm API Reference

## 1. Overview

This document provides a comprehensive reference for the APIs, data structures, and conventions required to develop custom collective communication algorithms for the Huawei Collective Communication Library (HCCL). It is synthesized from the official HCCL documentation and is intended to be the single source of truth for the HCCLang transpiler project.

---

## 2. Core Development Workflow

The development of a custom algorithm follows a strict lifecycle managed by the HCCL framework:

1. **Algorithm Selection (`SelectAlg`)**: The framework first calls this function on an `Operator` class. Based on the operation parameters (like data size, topology), this function returns the name of the specific `Executor` to use.
2. **Resource Calculation (`CalcResRequest`)**: Once the executor is chosen, the framework calls this function. The executor calculates all necessary resources (e.g., number of streams, scratch buffer size, communication links) and returns them in an `AlgResourceRequest` struct.
3. **Resource Creation**: The framework receives the request and allocates the necessary resources (streams, memory, transport links).
4. **Algorithm Orchestration (`Orchestrate`/`KernelRun`)**: The framework calls the final execution function, passing the allocated resources. This is where the core algorithm logic, which sends and receives data, is implemented.

---

## 3. Key Data Structures

These are the primary data structures used to pass information between the framework and the custom algorithm.

| Struct / Class          | Purpose                                                                                   | Key Fields / Methods                                                                                                |
| :---------------------- | :---------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------ |
| `OpParam`             | Contains all input parameters for a collective operation.                                 | `inputPtr`, `outputPtr`, `DataDes.count`, `DataDes.dataType`, `reduceType`, `stream`                    |
| `AlgResourceRequest`  | Describes the resources an algorithm needs to run.                                        | `scratchMemSize`, `streamNum`, `notifyNum`, `opTransport` (for link requests)                               |
| `AlgResourceResponse` | Contains the resources allocated by the framework.                                        | `cclInputMem`, `cclOutputMem`, `scratchMem`, `slaveStreams`, `notifiesM2S`, `opTransportResponse.links` |
| `DeviceMem`           | A wrapper for device memory.                                                              | `ptr()`, `size()`, `range(offset, size)`                                                                      |
| `Stream`              | Represents a CUDA/rtStream for enqueuing operations.                                      | `ptr()`, `id()`                                                                                                 |
| `LINK`                | A `std::shared_ptr<Transport>`, representing a communication channel between two ranks. | All communication primitives are called on this object.                                                             |

---

## 4. API Function Reference

### 4.1. Communication Primitives (on `LINK` object)

These are the fundamental operations for moving data between ranks.

| Function               | Signature                                                                                              | Purpose                                                                       |
| :--------------------- | :----------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------- |
| **TxAsync**      | `TxAsync(UserMemType, u64 offset, void* src, u64 len, Stream&)`                                      | Asynchronously sends data from a local buffer (`src`) to the peer's buffer. |
| **RxAsync**      | `RxAsync(UserMemType, u64 offset, void* dst, u64 len, Stream&)`                                      | Asynchronously receives data from a peer into a local buffer (`dst`).       |
| **TxWithReduce** | `TxWithReduce(UserMemType, u64, void*, u64, HcclDataType, HcclReduceOp, Stream&)`                | Fused operation that sends data and triggers a reduction on the remote end.   |
| **RxWithReduce** | `RxWithReduce(UserMemType, u64, void*, u64, void*, void*, u64, HcclDataType, HcclReduceOp, Stream&, u64)` | Fused operation that receives data and performs a reduction with local data.  |
| **TxData**       | `TxData(UserMemType, u64, void*, u64, Stream&)`                                                     | Synchronous data transmission with built-in synchronization.                 |
| **RxData**       | `RxData(UserMemType, u64, void*, u64, Stream&)`                                                     | Synchronous data reception with built-in synchronization.                    |
| **TxPrepare**    | `TxPrepare(Stream&)`                                                                                 | Prepares the link for transmission.                                          |
| **RxPrepare**    | `RxPrepare(Stream&)`                                                                                 | Prepares the link for reception.                                             |
| **TxDone**       | `TxDone(Stream&)`                                                                                    | Signals transmission completion.                                              |
| **RxDone**       | `RxDone(Stream&)`                                                                                    | Signals reception completion.                                                 |
| **Write**        | `Write(const void *localAddr, UserMemType remoteMemType, u64 remoteOffset, u64 len, Stream &stream)` | One-sided write operation.                                                    |
| **Read**         | `Read(const void *localAddr, UserMemType remoteMemType, u64 remoteOffset, u64 len, Stream &stream)`  | One-sided read operation.                                                     |

### 4.2. Synchronization Primitives

These are essential for ensuring correct ordering and data readiness.

| Function                 | Signature                                   | Purpose                                                                                                   |
| :----------------------- | :------------------------------------------ | :-------------------------------------------------------------------------------------------------------- |
| **TxAck**          | `TxAck(Stream&)`                          | **Readiness Handshake (Sender)**: Notifies the receiving peer that the sender is about to transmit. |
| **RxAck**          | `RxAck(Stream&)`                          | **Readiness Handshake (Receiver)**: Waits for the `TxAck` signal from the sending peer.           |
| **TxWaitDone**     | `TxWaitDone(Stream&)`                     | **Completion Sync (Sender)**: Waits until the corresponding `TxAsync` operation has completed.    |
| **RxWaitDone**     | `RxWaitDone(Stream&)`                     | **Completion Sync (Receiver)**: Waits until the corresponding `RxAsync` operation has completed.  |
| **ExecuteBarrier** | `ExecuteBarrier(linkLeft, linkRight)`     | A higher-level barrier that ensures all ranks have reached a certain point.                               |
| **Wait** (LocalNotify) | `Wait(Stream&, HcclDispatcher, s32, u32)` | Generic wait on a notify object for cross-stream synchronization.             |
| **Post** (LocalNotify) | `Post(Stream&, HcclDispatcher, s32)`      | Generic post to a notify object for cross-stream synchronization.             |
| **DataReceivedAck** | `DataReceivedAck(Stream&)`               | Acknowledges data reception completion.                                        |
| **PostReady**      | `PostReady(Stream&)`                      | Posts ready signal to notify remote peer.                                     |
| **WaitReady**      | `WaitReady(Stream&)`                      | Waits for ready signal from remote peer.                                      |
| **PostFin**        | `PostFin(Stream&)`                        | Posts finish signal.                                                           |
| **WaitFin**        | `WaitFin(Stream&)`                        | Waits for finish signal.                                                       |
| **PostFinAck**     | `PostFinAck(Stream&)`                     | Posts finish acknowledgment.                                                   |
| **WaitFinAck**     | `WaitFinAck(Stream&)`                     | Waits for finish acknowledgment.                                               |

### 4.3. Memory and Stream Operations

| Function                     | Signature                                                        | Purpose                                                                                                                      |
| :--------------------------- | :--------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------- |
| **HcclD2DMemcpyAsync** | `HcclD2DMemcpyAsync(dispatcher, dst, src, stream)`             | Asynchronously copies data between two locations on the same device.                                                         |
| **DeviceMem::range**   | `range(u64 offset, u64 size)`                                  | Creates a new `DeviceMem` object representing a sub-slice of an existing one. This is critical for chunk-based algorithms. |
| **alloc (DeviceMem)**  | `static DeviceMem alloc(u64 size, bool level2Address = false)` | Allocates device memory.                                                                                                     |
| **create (DeviceMem)** | `static DeviceMem create(void *ptr, u64 size)`                 | Creates a `DeviceMem` object from an existing pointer.                                                                     |
| **alloc (HostMem)**    | `static HostMem alloc(u64 size, bool isRtsMem = true)`         | Allocates host memory.                                                                                                       |
| **create (HostMem)**   | `static HostMem create(void *ptr, u64 size)`                   | Creates a `HostMem` object from an existing pointer.                                                                       |
| **GetSqeContext**      | `GetSqeContext(std::shared_ptr<HcclSqeContext> &sqeContext)`   | Gets the SqeContext from a stream.                                           |
| **GetStreamInfo**      | `GetStreamInfo(const HcclComStreamInfo*&)`                     | Gets stream information.                                                      |
| **IsMainStream**       | `IsMainStream()`                                                | Checks if the current stream is the main stream.                             |
| **LaunchTask**         | `LaunchTask(HcclDispatcher, Stream&)`                          | Launches a task on the specified stream.                                     |
| **HcclMemcpyAsync**    | `HcclMemcpyAsync(HcclDispatcher, void*, u64, void*, u64, HcclRtMemcpyKind, Stream&, u32, LinkType)` | Asynchronous memory copy with rank and link type specification.              |
| **HcclReduceAsync**    | `HcclReduceAsync(HcclDispatcher, void*, u64, HcclDataType, HcclReduceOp, Stream&, void*, u32, LinkType, u64)` | Asynchronous reduction operation.                                            |

---

## 5. Naming Conventions & Registration

Strict adherence to these conventions is mandatory for the HCCL framework to discover and use the custom algorithm.

### 5.1. File and Class Naming

* **Executor Header**: `coll_{collective_lower}_{topology_lower}_executor.h` (e.g., `coll_allgather_ring_executor.h`)
* **Executor Source**: `coll_{collective_lower}_{topology_lower}_executor.cc` (e.g., `coll_allgather_ring_executor.cc`)
* **Algorithm Header**: `{collective_lower}_{topology_lower}.h` (e.g., `allgather_ring.h`)
* **Algorithm Source**: `{collective_lower}_{topology_lower}.cc` (e.g., `allgather_ring.cc`)
* **Executor Class**: `Coll{CollectiveCamel}{TopologyCamel}Executor` (e.g., `CollAllgatherRingExecutor`)
* **Algorithm Class**: `{CollectiveCamel}{TopologyCamel}` (e.g., `AllgatherRing`)

### 5.2. Registration Macros

These macros connect the custom classes to the HCCL framework's factory pattern.

* **`REGISTER_EXEC("AllGatherComm", AllGatherComm, CollAllGatherCommExecutor);`**

  * **Location**: End of the executor `.cc` file.
  * **Purpose**: Registers the `Executor` to handle a specific collective (`Allgather`) with a specific topology (`Ring`).
* **`REGISTER_OP(HCCL_CMD_ALLGATHER, AllGatherOperator);`**

  * **Location**: End of the operator `.cc` file.
  * **Purpose**: Registers the `Operator` for a given HCCL command type.
