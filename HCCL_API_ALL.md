# HCCL API Complete Reference

This document provides a comprehensive reference for all HCCL APIs.
Generated automatically from the official HCCL documentation.

---

## Accept

**Description:** 发起建链请求。

**Prototype:**

```cpp
HcclResult Accept(const std::string &tag, std::shared_ptr<HcclSocket> &socket)
```

**Parameters:**

| Parameter                                | Direction | Description          |
| ---------------------------------------- | --------- | -------------------- |
| const std::string &tag                   | 输入      | Tag标识              |
| std::shared_ptr `<HcclSocket>` &socket | 输出      | 建链完成的socket对象 |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

---

## AddWhiteList

**Description:** 添加白名单。

**Prototype:**

```cpp
HcclResult AddWhiteList(std::vector<SocketWlistInfo> &wlistInfoVec)
```

**Parameters:**

| Parameter                                      | Direction | Description |
| ---------------------------------------------- | --------- | ----------- |
| std::vector`<SocketWlistInfo>` &wlistInfoVec | 输入      | 白名单信息  |
| 输入                                           |           | 白名单信息  |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

---

## AlgResourceRequest

**Description:** AlgResourceRequest结构体用于承载executor执行需要的资源诉求，包含从流数量、主从流同步需要的notify数量、Scratch Buffer、建链诉求等信息，由通信算法层计算并赋值。

**Prototype:**

```cpp
struct AlgResourceRequest {
u64 scratchMemSize = 0;
u32 streamNum = 0;
u32 notifyNum = 0;
bool needAivBuffer = false;
DeviceMode mode = DeviceMode::HOST;
OpCommTransport opTransport;
void Describe()
{
HCCL_DEBUG("[AlgResourceRequest], scratchMemSize[%u], streamNum[%u], notifyNum[%u], needAivBuffer[%u], "
"DeviceMode[%d].", scratchMemSize, streamNum, notifyNum, needAivBuffer, mode);
};
};
```

**Parameters:**

| Parameter       | Direction       | Description                                                        |
| --------------- | --------------- | ------------------------------------------------------------------ |
| scratchMemSize  | u64             | Executor执行需要的Scratch Buffer大小，用于暂存算法运行的中间结果。 |
| u64             |                 | Executor执行需要的Scratch Buffer大小，用于暂存算法运行的中间结果。 |
| streamNum       | u32             | Executor执行需要的从流数量。                                       |
| u32             |                 | Executor执行需要的从流数量。                                       |
| notifyNum       | u32             | 主从流同步需要的notify数量。                                       |
| u32             |                 | 主从流同步需要的notify数量。                                       |
| mode            | DeviceMode      | 用于区分是Host模式，还是AI CPU模式。                               |
| DeviceMode      |                 | 用于区分是Host模式，还是AI CPU模式。                               |
| opTransport     | OpCommTransport | 表示Executor执行需要的建链关系。                                   |
| OpCommTransport |                 | 表示Executor执行需要的建链关系。                                   |

---

## AlgResourceResponse

**Description:** AlgResourceResponse结构体用于存储资源创建的结果，由通信框架层创建并赋值。

**Prototype:**

```cpp
struct AlgResourceResponse {
DeviceMem cclInputMem;
DeviceMem cclOutputMem;
DeviceMem paramInputMem;
DeviceMem paramOutputMem;
DeviceMem scratchMem;
DeviceMem aivInputMem;
DeviceMem aivOutputMem;
std::vector<Stream> slaveStreams;
std::vector<Stream> slaveDevStreams;
std::vector<std::shared_ptr<LocalNotify> > notifiesM2S;  // 大小等同于slaveStreams
std::vector<std::shared_ptr<LocalNotify> > notifiesS2M;  // 大小等同于slaveStreams
std::vector<std::shared_ptr<LocalNotify> > notifiesDevM2S;  // 大小等同于slaveStreams
std::vector<std::shared_ptr<LocalNotify> > notifiesDevS2M;  // 大小等同于slaveStreams
OpCommTransport opTransportResponse;
};
```

**Parameters:**

| Parameter           | Direction      | Description                                                              |
| ------------------- | -------------- | ------------------------------------------------------------------------ |
| cclInputMem         | 内存对象       | 和通信域绑定的一块Device内存，单算子模式下可用于建链，通常用于缓存输入。 |
| 内存对象            |                | 和通信域绑定的一块Device内存，单算子模式下可用于建链，通常用于缓存输入。 |
| cclOutputMem        | 内存对象       | 和通信域绑定的一块Device内存，单算子模式下可用于建链，通常用于缓存输出。 |
| 内存对象            |                | 和通信域绑定的一块Device内存，单算子模式下可用于建链，通常用于缓存输出。 |
| paramInputMem       | 内存对象       | 算子的输入Device内存，图模式下可用于建链。                               |
| 内存对象            |                | 算子的输入Device内存，图模式下可用于建链。                               |
| paramOutputMem      | 内存对象       | 算子的输出Device内存，图模式下可用于建链。                               |
| 内存对象            |                | 算子的输出Device内存，图模式下可用于建链。                               |
| scratchMem          | 内存对象       | 算子的workspace内存，单算子或图模式下均可能使用，可用于建链。            |
| 内存对象            |                | 算子的workspace内存，单算子或图模式下均可能使用，可用于建链。            |
| aivInputMem         | 内存对象       | 算子的workspace内存，仅aiv场景使用。                                     |
| 内存对象            |                | 算子的workspace内存，仅aiv场景使用。                                     |
| aivOutputMem        | 内存对象       | 算子的workspace内存，仅aiv场景使用。                                     |
| 内存对象            |                | 算子的workspace内存，仅aiv场景使用。                                     |
| slaveStreams        | 流对象列表     | 算子需要的从流stream对象。                                               |
| 流对象列表          |                | 算子需要的从流stream对象。                                               |
| slaveDevStreams     | 流对象列表     | aicpu展开模式下，算子需要的从流stream对象。                              |
| 流对象列表          |                | aicpu展开模式下，算子需要的从流stream对象。                              |
| notifiesM2S         | notify对象列表 | 算子主流通知从流需要的notify资源。                                       |
| notify对象列表      |                | 算子主流通知从流需要的notify资源。                                       |
| notifiesS2M         | notify对象列表 | 算子从流通知主流需要的notify资源。                                       |
| notify对象列表      |                | 算子从流通知主流需要的notify资源。                                       |
| notifiesDevM2S      | notify对象列表 | aicpu展开模式下，算子主流通知从流需要的notify资源。                      |
| notify对象列表      |                | aicpu展开模式下，算子主流通知从流需要的notify资源。                      |
| notifiesDevS2M      | notify对象列表 | aicpu展开模式下，算子从流通知主流需要的notify资源。                      |
| notify对象列表      |                | aicpu展开模式下，算子从流通知主流需要的notify资源。                      |
| opTransportResponse | 建链表示结构体 | 和建链诉求是同一个结构体，可通过里面的links字段获取建好的链路。          |
| 建链表示结构体      |                | 和建链诉求是同一个结构体，可通过里面的links字段获取建好的链路。          |

---

## Break

**Description:** 退出。

**Prototype:**

```cpp
void Break()
```

**Return Value:** 无。

---

## ClearLocalBuff

**Description:** 清空sqe context里的buffer信息。

**Prototype:**

```cpp
HcclResult ClearLocalBuff()
```

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

---

## Close

**Description:** Socket断链。

**Prototype:**

```cpp
void Close()
```

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

---

## Connect

**Description:** Socket建链。

**Prototype:**

```cpp
HcclResult Connect()
```

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

---

## ConnectAsync

**Description:** 异步建链。

**Prototype:**

```cpp
HcclResult ConnectAsync(u32& status)
```

**Parameters:**

| Parameter   | Direction | Description |
| ----------- | --------- | ----------- |
| u32& status | 输出      | 建链状态    |
| 输出        |           | 建链状态    |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

---

## ConnectQuerry

**Description:** 建链结果查询。

**Prototype:**

```cpp
HcclResult ConnectQuerry(u32& status)
```

**Parameters:**

| Parameter   | Direction | Description |
| ----------- | --------- | ----------- |
| u32& status | 输出      | 建链状态    |
| 输出        |           | 建链状态    |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

---

## DataReceivedAck

**Description:** 接收数据后，发送同步信号到对端。

**Prototype:**

```cpp
HcclResult DataReceivedAck(Stream &stream)
```

**Parameters:**

| Parameter      | Direction | Description |
| -------------- | --------- | ----------- |
| Stream &stream | 输入      | Stream对象  |
| 输入           |           | Stream对象  |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

---

## DeInit

**Description:** Transport销毁。

**Prototype:**

```cpp
HcclResult DeInit()
```

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

---

## DeInit

**Description:** Socket销毁。

**Prototype:**

```cpp
HcclResult DeInit()
```

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

---

## DelWhiteList

**Description:** 删除白名单。

**Prototype:**

```cpp
HcclResult DelWhiteList(std::vector<SocketWlistInfo> &wlistInfoVec)
```

**Parameters:**

| Parameter                                      | Direction | Description |
| ---------------------------------------------- | --------- | ----------- |
| std::vector`<SocketWlistInfo>` &wlistInfoVec | 输入      | 白名单信息  |
| 输入                                           |           | 白名单信息  |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

---

## Destroy

**Description:** 销毁notify。

**Prototype:**

```cpp
virtual HcclResult Destroy()
```

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

---

## DeviceMem

**Description:** DeviceMem构造函数。

**Prototype:**

```cpp
// DeviceMem构造函数
DeviceMem()
DeviceMem(void *ptr, u64 size, bool owner = false)
//DevceMem 拷贝构造函数
DeviceMem(const DeviceMem &that)
// DeviceMem移动构造函数
DeviceMem(DeviceMem &&that)
```

**Parameters:**

| Parameter  | Direction | Description      |
| ---------- | --------- | ---------------- |
| void *ptr  | 输入      | 内存地址         |
| 输入       |           | 内存地址         |
| u64 size   | 输入      | 内存大小         |
| 输入       |           | 内存大小         |
| bool owner | 输入      | 是否是资源拥有者 |
| 输入       |           | 是否是资源拥有者 |

**Return Value:** 无。

---

## EnableUseOneDoorbell

**Description:** 使能一次DB功能。

**Prototype:**

```cpp
void EnableUseOneDoorbell()
```

**Return Value:** 无。

---

## GetBinaryAddress

**Description:** 获取IP信息（结构体）。

**Prototype:**

```cpp
union HcclInAddr GetBinaryAddress() const
```

**Return Value:** IP信息。

---

## GetFamily

**Description:** 获取IP地址族。

**Prototype:**

```cpp
s32 GetFamily() const
```

**Return Value:** IP地址族。

---

## GetFdHandle

**Description:** 获取FdHandle。

**Prototype:**

```cpp
FdHandle GetFdHandle() const
```

**Return Value:** FdHandle。

---

## GetIfName

**Description:** 获取网卡名。

**Prototype:**

```cpp
std::string GetIfName() const
```

**Return Value:** 网卡名。

---

## GetLinkType

**Description:** 获取链路类型。

**Prototype:**

```cpp
hccl::LinkType GetLinkType() const
```

**Return Value:** 链路类型。

---

## GetLocalIp

**Description:** 获取本端IP信息。

**Prototype:**

```cpp
HcclIpAddress GetLocalIp() const
```

**Return Value:** 本地IP信息。

---

## GetLocalPort

**Description:** 获取本端port。

**Prototype:**

```cpp
u32 GetLocalPort() const
```

**Return Value:** 本地port。

---

## GetLocalRole

**Description:** 获取本端在socket链接中的角色（server/client）。

**Prototype:**

```cpp
HcclSocketRole GetLocalRole() const
```

**Return Value:** 本端在socket链接中的角色。

---

## GetMode

**Description:** 获取logic sq id。

**Prototype:**

```cpp
u32 logicCqId() const
```

**Return Value:** Logic sq id。

---

## GetMode

**Description:** 获取stream模式。

**Prototype:**

```cpp
HcclResult GetMode(uint64_t *const stmMode)
```

**Parameters:**

| Parameter               | Direction | Description    |
| ----------------------- | --------- | -------------- |
| uint64_t *const stmMode | 输出      | 获取stream模式 |
| 输出                    |           | 获取stream模式 |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

---

## GetNextSqeBufferAddr

**Description:** 获取sqebuffer。

**Prototype:**

```cpp
HcclResult GetNextSqeBufferAddr(uint8_t *&sqeBufferAddr, uint8_t *&sqeTypeAddr, uint16_t &taskId)
```

**Parameters:**

| Parameter               | Direction | Description   |
| ----------------------- | --------- | ------------- |
| uint8_t *&sqeBufferAddr | 输出      | sqeBuffer地址 |
| 输出                    |           | sqeBuffer地址 |
| uint8_t *&sqeTypeAddr   | 输出      | sqeType地址   |
| 输出                    |           | sqeType地址   |
| uint16_t &taskId        | 输出      | Task id       |
| 输出                    |           | Task id       |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

---

## GetNotifyData

**Description:** 获取notify信息。

**Prototype:**

```cpp
HcclResult GetNotifyData(HcclSignalInfo &notifyInfo)
```

**Parameters:**

| Parameter                  | Direction | Description |
| -------------------------- | --------- | ----------- |
| HcclSignalInfo &notifyInfo | 输除      | Notify信息  |
| 输除                       |           | Notify信息  |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

---

## GetNotifyOffset

**Description:** 获取notify offset。

**Prototype:**

```cpp
HcclResult GetNotifyOffset(u64 &notifyOffset)
```

**Parameters:**

| Parameter         | Direction | Description   |
| ----------------- | --------- | ------------- |
| u64 &notifyOffset | 输出      | Notify offset |
| 输出              |           | Notify offset |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

---

## GetReadableAddress

**Description:** 获取IP地址（点分格式）+网卡名。

**Prototype:**

```cpp
const char *GetReadableAddress() const
```

**Return Value:** IP地址+网卡名。

---

## GetReadableIP

**Description:** 获取IP信息（点分格式）。

**Prototype:**

```cpp
const char *GetReadableIP() const
```

**Return Value:** IP信息。

---

## GetRemoteIp

**Description:** 获取对端IP。

**Prototype:**

```cpp
HcclIpAddress GetRemoteIp() const
```

**Return Value:** 对端IP信息。

---

## GetRemoteMem

**Description:** 获取远端交换的mem。

**Prototype:**

```cpp
HcclResult GetRemoteMem(UserMemType memType, void **remotePtr)
```

**Parameters:**

| Parameter           | Direction | Description  |
| ------------------- | --------- | ------------ |
| UserMemType memType | 输入      | 用户内存类型 |
| 输入                |           | 用户内存类型 |
| void **remotePtr    | 输出      | 对端内存地址 |
| 输出                |           | 对端内存地址 |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

---

## GetRemotePort

**Description:** 获取对端port。

**Prototype:**

```cpp
u32 GetRemotePort() const
```

**Return Value:** 对端port。

---

## GetRemoteRank

**Description:** 获取对端user rank。

**Prototype:**

```cpp
u32 GetRemoteRank()
```

**Return Value:** 对端user rank。

---

## GetRxAckDevNotifyInfo

**Description:** 获取同步notify信息相关信息。

**Prototype:**

```cpp
HcclResult GetRxAckDevNotifyInfo(HcclSignalInfo &notifyInfo)
```

**Parameters:**

| Parameter                  | Direction | Description |
| -------------------------- | --------- | ----------- |
| HcclSignalInfo &notifyInfo | 输出      | Notify信息  |
| 输出                       |           | Notify信息  |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

---

## GetRxDataSigleDevNotifyInfo

**Description:** 获取同步notify信息相关信息。

**Prototype:**

```cpp
HcclResult GetRxDataSigleDevNotifyInfo(HcclSignalInfo &notifyInfo)
```

**Parameters:**

| Parameter                  | Direction | Description |
| -------------------------- | --------- | ----------- |
| HcclSignalInfo &notifyInfo | 输出      | Notify信息  |
| 输出                       |           | Notify信息  |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

---

## GetScopeID

**Description:** 获取Ip的ScopeID。

**Prototype:**

```cpp
s32 GetScopeID() const
```

**Return Value:** Ip的ScopeID。

---

## GetSocketType

**Description:** 获取网卡类型。

**Prototype:**

```cpp
NicType GetSocketType() const
```

**Return Value:** 网卡类型。

---

## GetSqeContext

**Description:** 获取sqe context。

**Prototype:**

```cpp
HcclResult GetSqeContext(std::shared_ptr<HcclSqeContext> &sqeContext)
```

**Parameters:**

| Parameter                                       | Direction | Description |
| ----------------------------------------------- | --------- | ----------- |
| std::shared_ptr`<HcclSqeContext>` &sqeContext | 输出      | Sqe context |
| 输出                                            |           | Sqe context |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

---

## GetStatus

**Description:** 获取socket建链状态。

**Prototype:**

```cpp
HcclSocketStatus GetStatus()
```

**Return Value:** Socket建链状态。

---

## GetStreamInfo

**Description:** 获取stream信息。

**Prototype:**

```cpp
HcclResult GetStreamInfo(const HcclComStreamInfo *&streamInfo)
```

**Parameters:**

| Parameter                            | Direction | Description |
| ------------------------------------ | --------- | ----------- |
| const HcclComStreamInfo *&streamInfo | 输出      | Stream信息  |
| 输出                                 |           | Stream信息  |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

---

## GetSupportDataReceivedAck

**Description:** 获取是否支持收到数据后返回同步信号功能。

**Prototype:**

```cpp
bool GetSupportDataReceivedAck() const
```

**Return Value:** False：不支持；true：支持。

---

## GetTag

**Description:** 获取socket tag。

**Prototype:**

```cpp
std::string GetTag() const
```

**Return Value:** Socket tag。

---

## GetTxAckDevNotifyInfo

**Description:** 获取同步notify信息相关信息。

**Prototype:**

```cpp
HcclResult GetTxAckDevNotifyInfo(HcclSignalInfo &notifyInfo)
```

**Parameters:**

| Parameter                  | Direction | Description |
| -------------------------- | --------- | ----------- |
| HcclSignalInfo &notifyInfo | 输出      | Notify信息  |
| 输出                       |           | Notify信息  |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

---

## GetTxDataSigleDevNotifyInfo

**Description:** 获取同步notify信息相关信息。

**Prototype:**

```cpp
HcclResult GetTxDataSigleDevNotifyInfo(HcclSignalInfo &notifyInfo)
```

**Parameters:**

| Parameter                  | Direction | Description |
| -------------------------- | --------- | ----------- |
| HcclSignalInfo &notifyInfo | 输出      | Notify信息  |
| 输出                       |           | Notify信息  |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

---

## GetUseOneDoorbellValue

**Description:** 获取是否使能一次DB功能标记。

**Prototype:**

```cpp
bool GetUseOneDoorbellValue()
```

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

---

## HcclD2DMemcpyAsync

**Description:** 异步device间内存copy。

**Prototype:**

```cpp
HcclResult HcclD2DMemcpyAsync(HcclDispatcher dispatcherPtr, hccl::DeviceMem &dst, const hccl::DeviceMem &src, hccl::Stream &stream, u32 remoteUserRank = INVALID_VALUE_RANKID, hccl::LinkType inLinkType = hccl::LinkType::LINK_ONCHIP)
```

**Parameters:**

| Parameter                  | Direction | Description       |
| -------------------------- | --------- | ----------------- |
| HcclDispatcher dispatcher  | 输入      | dispatcher handle |
| 输入                       |           | dispatcher handle |
| hccl::DeviceMem &dst       | 输入      | dst内存对象       |
| 输入                       |           | dst内存对象       |
| const hccl::DeviceMem &src | 输入      | src内存对象       |
| 输入                       |           | src内存对象       |
| hccl::Stream &stream       | 输入      | stream对象        |
| 输入                       |           | stream对象        |
| u32 remoteUserRank         | 输入      | 对端world rank    |
| 输入                       |           | 对端world rank    |
| hccl::LinkType inLinkType  | 输入      | 链路类型          |
| 输入                       |           | 链路类型          |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS；其他失败。

---

## HcclDispatcherDestroy

**Description:** 销毁dispatcher。

**Prototype:**

```cpp
HcclResult HcclDispatcherDestroy(HcclDispatcher dispatcher)
```

**Parameters:**

| Parameter                 | Direction | Description       |
| ------------------------- | --------- | ----------------- |
| HcclDispatcher dispatcher | 输入      | dispatcher handle |
| 输入                      |           | dispatcher handle |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS；其他失败。

---

## HcclDispatcherInit

**Description:** 初始化dispatcher。

**Prototype:**

```cpp
HcclResult HcclDispatcherInit(DispatcherType type, const s32 deviceLogicId, const std::shared_ptr<hccl::ProfilerManager> &profilerManager, HcclDispatcher *dispatcher)
```

**Parameters:**

| Parameter                                                                         | Direction | Description       |
| --------------------------------------------------------------------------------- | --------- | ----------------- |
| DispatcherType type                                                               | 输入      | dispatcher 类型   |
| 输入                                                                              |           | dispatcher 类型   |
| const s32 deviceLogicId                                                           | 输入      | deviceLogicId     |
| 输入                                                                              |           | deviceLogicId     |
| const std::shared_ptr[hccl::ProfilerManager](hccl::ProfilerManager) &profilerManager | 输入      | ProfilerManager   |
| 输入                                                                              |           | ProfilerManager   |
| HcclDispatcher *dispatcher                                                        | 输出      | dispatcher handle |
| 输出                                                                              |           | dispatcher handle |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS；其他失败。

---

## HcclGetCallbackResult

**Description:** 获取callback执行结果。

**Prototype:**

```cpp
HcclResult HcclGetCallbackResult(HcclDispatcher dispatcherPtr)
```

**Parameters:**

| Parameter                 | Direction | Description       |
| ------------------------- | --------- | ----------------- |
| HcclDispatcher dispatcher | 输入      | dispatcher handle |
| 输入                      |           | dispatcher handle |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS；其他失败。

---

## HcclGetNotifyWaitMode

**Description:** 获取notify wait工作模式。

**Prototype:**

```cpp
HcclResult HcclGetNotifyWaitMode(HcclDispatcher dispatcherPtr, SyncMode *notifyWaitMode)
```

**Parameters:**

| Parameter                 | Direction | Description         |
| ------------------------- | --------- | ------------------- |
| HcclDispatcher dispatcher | 输入      | dispatcher handle   |
| 输入                      |           | dispatcher handle   |
| SyncMode *notifyWaitMode  | 输出      | notify wait工作模式 |
| 输出                      |           | notify wait工作模式 |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS；其他失败。

---

## HcclGetQosCfg

**Description:** 获取qos cfg。

**Prototype:**

```cpp
HcclResult HcclGetQosCfg(HcclDispatcher dispatcherPtr, u32 *qosCfg)
```

**Parameters:**

| Parameter                 | Direction | Description       |
| ------------------------- | --------- | ----------------- |
| HcclDispatcher dispatcher | 输入      | dispatcher handle |
| 输入                      |           | dispatcher handle |
| u32 *qosCfg               | 输出      | qos cfg           |
| 输出                      |           | qos cfg           |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS；其他失败。

---

## HcclIpAddress

**Description:** 构造IpAddress。

**Prototype:**

```cpp
HcclIpAddress()
HcclIpAddress(u32 address)
HcclIpAddress(s32 family, const union HcclInAddr &address)
HcclIpAddress(const struct in_addr &address)
HcclIpAddress(const struct in6_addr &address)
HcclIpAddress(const std::string &address)
```

**Parameters:**

| Parameter                       | Direction | Description        |
| ------------------------------- | --------- | ------------------ |
| u32 address                     | 输入      | U32表示的Ip        |
| 输入                            |           | U32表示的Ip        |
| s32 family                      | 输入      | Ip地址族           |
| 输入                            |           | Ip地址族           |
| const union HcclInAddr &address | 输入      | Ip信息             |
| 输入                            |           | Ip信息             |
| const struct in_addr &address   | 输入      | Ipv4信息           |
| 输入                            |           | Ipv4信息           |
| const struct in6_addr &address  | 输入      | Ipv6信息           |
| 输入                            |           | Ipv6信息           |
| const std::string &address      | 输入      | 字符串类型的Ip地址 |
| 输入                            |           | 字符串类型的Ip地址 |

**Return Value:** 无。

---

## \~HcclIpAddress

**Description:** IpAddress析构函数。

**Return Value:** 无。

---

## HcclMemcpyAsync

**Description:** 异步内存copy。

**Prototype:**

```cpp
HcclResult HcclMemcpyAsync(HcclDispatcher dispatcherPtr, void *dst, const uint64_t destMax, const void *src, const uint64_t count, const HcclRtMemcpyKind kind, hccl::Stream &stream, const u32 remoteUserRank, hccl::LinkType linkType)
```

**Parameters:**

| Parameter                   | Direction | Description       |
| --------------------------- | --------- | ----------------- |
| HcclDispatcher dispatcher   | 输入      | dispatcher handle |
| 输入                        |           | dispatcher handle |
| void *dst                   | 输入      | dst内存地址       |
| 输入                        |           | dst内存地址       |
| const uint64_t destMax      | 输入      | dst内存大小       |
| 输入                        |           | dst内存大小       |
| const void *src             | 输入      | src内存地址       |
| 输入                        |           | src内存地址       |
| const uint64_t count        | 输入      | src内存大小       |
| 输入                        |           | src内存大小       |
| const HcclRtMemcpyKind kind | 输入      | 内存copy类型      |
| 输入                        |           | 内存copy类型      |
| hccl::Stream &stream        | 输入      | stream对象        |
| 输入                        |           | stream对象        |
| u32 remoteUserRank          | 输入      | 对端world rank    |
| 输入                        |           | 对端world rank    |
| hccl::LinkType inLinkType   | 输入      | 链路类型          |
| 输入                        |           | 链路类型          |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS；其他失败。

---

## HcclNetCloseDev

**Description:** 关闭网卡设备。

**Prototype:**

```cpp
void HcclNetCloseDev(HcclNetDevCtx netDevCtx)
```

**Parameters:**

| Parameter                | Direction | Description    |
| ------------------------ | --------- | -------------- |
| HcclNetDevCtx *netDevCtx | 输入      | 网卡设备handle |
| 输入                     |           | 网卡设备handle |

**Return Value:** 无。

---

## HcclNetDeInit

**Description:** 销毁网络功能。

**Prototype:**

```cpp
HcclResult HcclNetDeInit(NICDeployment nicDeploy, s32 devicePhyId, s32 deviceLogicId)
```

**Parameters:**

| Parameter               | Direction | Description     |
| ----------------------- | --------- | --------------- |
| NICDeployment nicDeploy | 输入      | 网卡部署位置    |
| 输入                    |           | 网卡部署位置    |
| s32 devicePhyId         | 输入      | Device phy ID   |
| 输入                    |           | Device phy ID   |
| s32 deviceLogicId       | 输入      | Device logic ID |
| 输入                    |           | Device logic ID |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

---

## HcclNetDevGetLocalIp

**Description:** 获取对应的网卡ip。

**Prototype:**

```cpp
HcclResult HcclNetDevGetLocalIp(HcclNetDevCtx netDevCtx, hccl::HcclIpAddress &localIp)
```

**Parameters:**

| Parameter                    | Direction | Description    |
| ---------------------------- | --------- | -------------- |
| HcclNetDevCtx *netDevCtx     | 输入      | 网卡设备handle |
| 输入                         |           | 网卡设备handle |
| hccl::HcclIpAddress &localIp | 输出      | Ip信息         |
| 输出                         |           | Ip信息         |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

---

## HcclNetDevGetNicType

**Description:** 获取网卡类型。

**Prototype:**

```cpp
HcclResult HcclNetDevGetNicType(HcclNetDevCtx netDevCtx, NicType *nicType)
```

**Parameters:**

| Parameter                | Direction | Description    |
| ------------------------ | --------- | -------------- |
| HcclNetDevCtx *netDevCtx | 输入      | 网卡设备handle |
| 输入                     |           | 网卡设备handle |
| NicType *nicType         | 输出      | 网卡类型       |
| 输出                     |           | 网卡类型       |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

---

## HcclNetInit

**Description:** 初始化网络功能。

**Prototype:**

```cpp
HcclResult HcclNetInit(NICDeployment nicDeploy, s32 devicePhyId, s32 deviceLogicId, bool enableWhitelistFlag)
```

**Parameters:**

| Parameter                | Direction | Description        |
| ------------------------ | --------- | ------------------ |
| NICDeployment nicDeploy  | 输入      | 网卡部署位置       |
| 输入                     |           | 网卡部署位置       |
| s32 devicePhyId          | 输入      | Device phy ID      |
| 输入                     |           | Device phy ID      |
| s32 deviceLogicId        | 输入      | Device logic ID    |
| 输入                     |           | Device logic ID    |
| bool enableWhitelistFlag | 输入      | 是否开启白名单校验 |
| 输入                     |           | 是否开启白名单校验 |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

---

## HcclNetOpenDev

**Description:** 打开网卡设备。

**Prototype:**

```cpp
HcclResult HcclNetOpenDev(HcclNetDevCtx *netDevCtx, NicType nicType, s32 devicePhyId, s32 deviceLogicId, hccl::HcclIpAddress localIp)
```

**Parameters:**

| Parameter                   | Direction | Description     |
| --------------------------- | --------- | --------------- |
| HcclNetDevCtx *netDevCtx    | 输出      | 网卡设备handle  |
| 输出                        |           | 网卡设备handle  |
| NicType nicType             | 输入      | 网卡类型        |
| 输入                        |           | 网卡类型        |
| s32 devicePhyId             | 输入      | Device phy ID   |
| 输入                        |           | Device phy ID   |
| s32 deviceLogicId           | 输入      | Device logic ID |
| 输入                        |           | Device logic ID |
| hccl::HcclIpAddress localIp | 输入      | 网卡ip信息      |
| 输入                        |           | 网卡ip信息      |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

---

## HcclReduceAsync

**Description:** 异步reduce。

**Prototype:**

```cpp
HcclResult HcclReduceAsync(HcclDispatcher dispatcherPtr, void *src, uint64_t count, const HcclDataType datatype, const HcclReduceOp reduceOp, hccl::Stream &stream, void *dst, const u32 remoteUserRank, const hccl::LinkType linkType, const u64 reduceAttr)
```

**Parameters:**

| Parameter                 | Direction | Description       |
| ------------------------- | --------- | ----------------- |
| HcclDispatcher dispatcher | 输入      | dispatcher handle |
| 输入                      |           | dispatcher handle |
| void *src                 | 输入      | src内存地址       |
| 输入                      |           | src内存地址       |
| uint64_t count            | 输入      | reduce mem大小    |
| 输入                      |           | reduce mem大小    |
| HcclDataType datatype     | 输入      | 数据类型          |
| 输入                      |           | 数据类型          |
| HcclReduceOp reduceOp     | 输入      | reduce op类型     |
| 输入                      |           | reduce op类型     |
| hccl::Stream &stream      | 输入      | stream对象        |
| 输入                      |           | stream对象        |
| void *dst                 | 输入      | dst内存地址       |
| 输入                      |           | dst内存地址       |
| u32 remoteUserRank        | 输入      | 对端world rank    |
| 输入                      |           | 对端world rank    |
| hccl::LinkType inLinkType | 输入      | 链路类型          |
| 输入                      |           | 链路类型          |
| const u64 reduceAttr      | 输入      | reduce类型        |
| 输入                      |           | reduce类型        |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS；其他失败。

---

## HcclResetQosCfg

**Description:** 重置qos cfg。

**Prototype:**

```cpp
HcclResult HcclResetQosCfg(HcclDispatcher dispatcherPtr)
```

**Parameters:**

| Parameter                 | Direction | Description       |
| ------------------------- | --------- | ----------------- |
| HcclDispatcher dispatcher | 输入      | dispatcher handle |
| 输入                      |           | dispatcher handle |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS；其他失败。

---

## HcclRtNotify ptr

**Description:** 获取notify ptr。

**Prototype:**

```cpp
inline HcclRtNotify ptr()
```

**Return Value:** Notify ptr。

---

## HcclSetGlobalWorkSpace

**Description:** 设置global workspace mem。

**Prototype:**

```cpp
HcclResult HcclSetGlobalWorkSpace(HcclDispatcher dispatcherPtr, std::vector<void *> &globalWorkSpaceAddr)
```

**Parameters:**

| Parameter                                | Direction | Description          |
| ---------------------------------------- | --------- | -------------------- |
| HcclDispatcher dispatcher                | 输入      | dispatcher handle    |
| 输入                                     |           | dispatcher handle    |
| std::vector<void *> &globalWorkSpaceAddr | 输出      | global workspace mem |
| 输出                                     |           | global workspace mem |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS；其他失败。

---

## HcclSetNotifyWaitMode

**Description:** 设置notify wait工作模式。

**Prototype:**

```cpp
HcclResult HcclSetNotifyWaitMode(HcclDispatcher dispatcherPtr, const SyncMode notifyWaitMode)
```

**Parameters:**

| Parameter                     | Direction | Description         |
| ----------------------------- | --------- | ------------------- |
| HcclDispatcher dispatcher     | 输入      | dispatcher handle   |
| 输入                          |           | dispatcher handle   |
| const SyncMode notifyWaitMode | 输入      | notify wait工作模式 |
| 输入                          |           | notify wait工作模式 |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS；其他失败。

---

## HcclSetQosCfg

**Description:** 设置qos cfg。

**Prototype:**

```cpp
HcclResult HcclSetQosCfg(HcclDispatcher dispatcherPtr, const u32 qosCfg)
```

**Parameters:**

| Parameter                 | Direction | Description       |
| ------------------------- | --------- | ----------------- |
| HcclDispatcher dispatcher | 输入      | dispatcher handle |
| 输入                      |           | dispatcher handle |
| const u32 qosCfg          | 输入      | qos cfg           |
| 输入                      |           | qos cfg           |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS；其他失败。

---

## HcclSignalRecord

**Description:** notify record。

**Prototype:**

```cpp
HcclResult HcclSignalRecord(HcclDispatcher dispatcherPtr, HcclRtNotify signal, hccl::Stream &stream, u32 userRank, u64 offset, s32 stage, bool inchip, u64 signalAddr)
```

**Parameters:**

| Parameter                 | Direction | Description       |
| ------------------------- | --------- | ----------------- |
| HcclDispatcher dispatcher | 输入      | dispatcher handle |
| 输入                      |           | dispatcher handle |
| HcclRtNotify signal       | 输入      | rt notify         |
| 输入                      |           | rt notify         |
| hccl::Stream &stream      | 输入      | stream对象        |
| 输入                      |           | stream对象        |
| u32 userRank              | 输入      | 本端world rank    |
| 输入                      |           | 本端world rank    |
| u64 offset                | 输入      | notify offset     |
| 输入                      |           | notify offset     |
| s32 stage                 | 输入      | 算法stage         |
| 输入                      |           | 算法stage         |
| bool inchip               | 输入      | 是否跨片          |
| 输入                      |           | 是否跨片          |
| u64 signalAddr            | 输入      | notify address    |
| 输入                      |           | notify address    |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS；其他失败。

---

## HcclSignalWait

**Description:** notify wait。

**Prototype:**

```cpp
HcclResult HcclSignalWait(HcclDispatcher dispatcherPtr, HcclRtNotify signal, hccl::Stream &stream, u32 userRank, u32 remoteUserRank, s32 stage, bool inchip)
```

**Parameters:**

| Parameter                 | Direction | Description       |
| ------------------------- | --------- | ----------------- |
| HcclDispatcher dispatcher | 输入      | dispatcher handle |
| 输入                      |           | dispatcher handle |
| HcclRtNotify signal       | 输入      | rt notify         |
| 输入                      |           | rt notify         |
| hccl::Stream &stream      | 输入      | stream对象        |
| 输入                      |           | stream对象        |
| u32 userRank              | 输入      | 本端world rank    |
| 输入                      |           | 本端world rank    |
| u32 remoteUserRank        | 输入      | 对端world rank    |
| 输入                      |           | 对端world rank    |
| s32 stage                 | 输入      | 算法stage         |
| 输入                      |           | 算法stage         |
| bool inchip               | 输入      | 是否跨片          |
| 输入                      |           | 是否跨片          |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS；其他失败。

---

## HcclSocket

**Description:** 构造socket对象。

**Prototype:**

```cpp
HcclSocket(const std::string &tag, HcclNetDevCtx netDevCtx, const HcclIpAddress &remoteIp, u32 remotePort, HcclSocketRole localRole)
HcclSocket(HcclNetDevCtx netDevCtx, u32 localPort)
```

**Parameters:**

| Parameter                     | Direction | Description    |
| ----------------------------- | --------- | -------------- |
| const std::string &tag        | 输入      | Tag标识        |
| 输入                          |           | Tag标识        |
| HcclNetDevCtx netDevCtx       | 输入      | 网卡设备handle |
| 输入                          |           | 网卡设备handle |
| const HcclIpAddress &remoteIp | 输入      | 对端IP信息     |
| 输入                          |           | 对端IP信息     |
| u32 remotePort                | 输入      | 对端prot       |
| 输入                          |           | 对端prot       |
| HcclSocketRole localRole      | 输入      | 本地建链角色   |
| 输入                          |           | 本地建链角色   |
| HcclNetDevCtx netDevCtx       | 输入      | 网卡设备handle |
| 输入                          |           | 网卡设备handle |
| u32 localPort                 | 输入      | 本端port       |
| 输入                          |           | 本端port       |

**Return Value:** 无。

---

## \~HcclSocket

**Description:** HcclSocket析构函数。

**Prototype:**

```cpp
~HcclSocket()
```

**Return Value:** 无。

---

## HostMem

**Description:** Host Mem构造函数。

**Prototype:**

```cpp
HostMem()
HostMem(const HostMem &that)
HostMem(HostMem &&that)
```

**Parameters:**

| Parameter     | Direction | Description |
| ------------- | --------- | ----------- |
| HostMem &that | 输入      | HostMem对象 |
| 输入          |           | HostMem对象 |

**Return Value:** 无。

---

## IRecv

**Description:** 非阻塞接收。

**Prototype:**

```cpp
HcclResult IRecv(void *recvBuf, u32 recvBufLen, u64& compSize)
```

**Parameters:**

| Parameter      | Direction | Description        |
| -------------- | --------- | ------------------ |
| void *recvBuf  | 输入      | 接收数据起始地址   |
| 输入           |           | 接收数据起始地址   |
| u32 recvBufLen | 输入      | 接收数据buffer大小 |
| 输入           |           | 接收数据buffer大小 |
| u64& compSize  | 输出      | 实际接收数据量     |
| 输出           |           | 实际接收数据量     |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

---

## ISend

**Description:** 非阻塞发送。

**Prototype:**

```cpp
HcclResult ISend(void *data, u64 size, u64& compSize)
```

**Parameters:**

| Parameter     | Direction | Description      |
| ------------- | --------- | ---------------- |
| void *data    | 输入      | 发送数据起始地址 |
| 输入          |           | 发送数据起始地址 |
| u64 size      | 输入      | 发送数据大小     |
| 输入          |           | 发送数据大小     |
| u64& compSize | 输出      | 实际发送数据大小 |
| 输出          |           | 实际发送数据大小 |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

---

## Init

**Description:** Notify初始化。

**Prototype:**

```cpp
HcclResult Init(const NotifyLoadType type = NotifyLoadType::HOST_NOTIFY)
HcclResult Init(const HcclSignalInfo &notifyInfo, const NotifyLoadType type = NotifyLoadType::DEVICE_NOTIFY)
```

**Parameters:**

| Parameter                        | Direction | Description    |
| -------------------------------- | --------- | -------------- |
| const NotifyLoadType type        | 输入      | Notify任务类型 |
| 输入                             |           | Notify任务类型 |
| const HcclSignalInfo &notifyInfo | 输入      | Notify信息     |
| 输入                             |           | Notify信息     |
| const NotifyLoadType type        | 输入      | Notify任务类型 |
| 输入                             |           | Notify任务类型 |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

---

## Init

**Description:** Transport初始化。

**Prototype:**

```cpp
HcclResult Init()
```

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

---

## Init

**Description:** Socket初始化。

**Prototype:**

```cpp
HcclResult Init()
```

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

---

## InitSqeContext

**Description:** 初始化SqeContext资源。

**Prototype:**

```cpp
HcclResult InitSqeContext(uint32_t sqHead, uint32_t sqTail)
```

**Parameters:**

| Parameter       | Direction | Description |
| --------------- | --------- | ----------- |
| uint32_t sqHead | 输入      | Sq head值   |
| 输入            |           | Sq head值   |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

---

## InitTask

**Description:** 初始化task。

**Prototype:**

```cpp
HcclDispatcher dispatcherPtr, hccl::Stream &stream, const hccl::HcclOpMetaInfo &opMetaInfo)
```

**Parameters:**

| Parameter                              | Direction | Description       |
| -------------------------------------- | --------- | ----------------- |
| HcclDispatcher dispatcher              | 输入      | dispatcher handle |
| 输入                                   |           | dispatcher handle |
| hccl::Stream &stream                   | 输入      | stream对象        |
| 输入                                   |           | stream对象        |
| const hccl::HcclOpMetaInfo &opMetaInfo | 输入      | opinfo            |
| 输入                                   |           | opinfo            |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS；其他失败。

---

## IsCtxInitialized

**Description:** task是否初始化。

**Prototype:**

```cpp
HcclResult IsCtxInitialized(HcclDispatcher dispatcherPtr, bool *ctxInitFlag)
```

**Parameters:**

| Parameter                 | Direction | Description       |
| ------------------------- | --------- | ----------------- |
| HcclDispatcher dispatcher | 输入      | dispatcher handle |
| 输入                      |           | dispatcher handle |
| bool *ctxInitFlag         | 输出      | 初始化标识        |
| 输出                      |           | 初始化标识        |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS；其他失败。

---

## IsIPv6

**Description:** 判断当前IP是否是IPV6。

**Prototype:**

```cpp
bool IsIPv6() const
```

**Return Value:** false：非IPV6；true：IPV6。

---

## IsInvalid

**Description:** 判断当前IP地址是否有效。

**Prototype:**

```cpp
bool IsInvalid() const
```

**Return Value:** false：非法；true：合法。

---

## IsMainStream

**Description:** 设置流的主、从流属性信息。

**Prototype:**

```cpp
inline bool IsMainStream()
```

**Return Value:** False：从流，true：主流。

---

## IsSpInlineReduce

**Description:** 是否支持inline reduce。

**Prototype:**

```cpp
bool IsSpInlineReduce() const
```

**Return Value:** False：不支持；true：支持。

---

## IsSupportTransportWithReduce

**Description:** 是否支持transport with reduce。

**Prototype:**

```cpp
bool IsSupportTransportWithReduce()
```

**Return Value:** false：不支持；true：支持。

---

## IsTransportRoce

**Description:** 是否支持transport roce。

**Prototype:**

```cpp
bool IsTransportRoce()
```

**Return Value:** False：不支持；true：支持。

---

## LaunchTask

**Description:** launch task。

**Prototype:**

```cpp
HcclResult LaunchTask(HcclDispatcher dispatcherPtr, hccl::Stream &stream)
```

**Parameters:**

| Parameter                 | Direction | Description       |
| ------------------------- | --------- | ----------------- |
| HcclDispatcher dispatcher | 输入      | dispatcher handle |
| 输入                      |           | dispatcher handle |
| hccl::Stream &stream      | 输入      | stream对象        |
| 输入                      |           | stream对象        |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS；其他失败。

---

## Listen

**Description:** Socket listen。

**Prototype:**

```cpp
HcclResult Listen()
```

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

---

## \~LocalNotify

**Description:** LocalNotify析构函数。

**Return Value:** 无。

---

## LocalNotify

**Description:** LocalNotify构造函数。

**Prototype:**

```cpp
LocalNotify()
```

**Return Value:** 无。

---

## OpParam

**Description:** OpParam结构体用于承载算子所有可能用到的入参信息。

**Prototype:**

```cpp
struct OpParam {
std::string tag = "";
Stream stream;
void* inputPtr = nullptr;
u64 inputSize = 0;
void* outputPtr = nullptr;
u64 outputSize = 0;
HcclReduceOp reduceType = HcclReduceOp::HCCL_REDUCE_RESERVED;
SyncMode syncMode = SyncMode::DEFAULT_TIMEWAITSYNCMODE;
RankId root = INVALID_VALUE_RANKID;
RankId dstRank = 0;
RankId srcRank = 0;
bool aicpuUnfoldMode = false;
HcclOpBaseAtraceInfo* opBaseAtraceInfo = nullptr;
union {
struct {
u64 count;
HcclDataType dataType;
} DataDes;
struct {
HcclDataType sendType;
HcclDataType recvType;
u64 sendCount;
void* sendCounts;
void* recvCounts;
void* sdispls;
void* rdispls;
void* sendCountMatrix;
} All2AllDataDes;
struct {
HcclSendRecvItem* sendRecvItemsPtr;
u32 itemNum;
} BatchSendRecvDataDes;
};
HcclCMDType opType = HcclCMDType::HCCL_CMD_INVALID;
};
```

**Parameters:**

| Parameter        | Direction | Description                                                                           |
| ---------------- | --------- | ------------------------------------------------------------------------------------- |
| tag              |           | 算子在通信域中的标记，用于维测功能。                                                  |
| stream           |           | 算子执行的主流。                                                                      |
| inputPtr         |           | 输入内存的指针，默认为nullptr。                                                       |
| inputSize        |           | 输入内存大小。                                                                        |
| outputPtr        |           | 输出内存的指针，默认为nullptr。                                                       |
| outputSize       |           | 输出内存大小。                                                                        |
| reduceType       |           | 消减运算类型，枚举值。                                                                |
| syncMode         |           | notifywait超时类型，默认为DEFAULT_TIMEWAITSYNCMODE。                                  |
| root             |           | root节点的rank id，默认值为INVALID_VALUE_RANKID，用于Reduce、Scatter和BroadCast算子。 |
| dstRank          |           | 目的rank id，用于Send/Recv算子。                                                      |
| srcRank          |           | 源rank id，用于Send/Recv算子。                                                        |
| aicpuUnfoldMode  |           | 是否为aicpu展开模式。                                                                 |
| opBaseAtraceInfo |           | Atrace管理类对象的指针，用于保存trace日志。                                           |
| union            |           | DataDes（通用定义）                                                                   |
| count            |           | 输入数据个数                                                                          |
| dataType         |           | 输入数据类型，如int8, in16, in32, float16, fload32等                                  |
| sendType         |           | 发送数据类型                                                                          |
| recvType         |           | 接收数据类型                                                                          |
| sendCounts       |           | 发送数据个数                                                                          |
| recvCounts       |           | 接收数据个数                                                                          |
| sdispls          |           | 表示发送偏移量的uint64数组                                                            |
| rdispls          |           | 表示接收偏移量的uint64数组                                                            |
| sendCountMatrix  |           | 代表每张卡要发给别人的count的信息                                                     |
| orderedList      |           | 发送和接收的item列表                                                                  |
| itemNum          |           | item数量                                                                              |
| opType           |           | 算子类型                                                                              |

---

## PopTaskLogicInfo

**Description:** 获取一个逻辑task信息。

**Prototype:**

```cpp
HcclResult PopTaskLogicInfo(TaskLogicInfo &taskLogicInfo)
```

**Parameters:**

| Parameter                    | Direction | Description |
| ---------------------------- | --------- | ----------- |
| TaskLogicInfo &taskLogicInfo | 输出      | Task信息    |
| 输出                         |           | Task信息    |

**Return Value:** 无。

---

## Post

**Description:** Notify post任务。

**Prototype:**

```cpp
HcclResult Post(Stream& stream, HcclDispatcher dispatcher, s32 stage = INVALID_VALUE_STAGE)
```

**Parameters:**

| Parameter                 | Direction | Description       |
| ------------------------- | --------- | ----------------- |
| Stream& stream            | 输入      | Stream对象        |
| 输入                      |           | Stream对象        |
| HcclDispatcher dispatcher | 输入      | Dispatcher handle |
| 输入                      |           | Dispatcher handle |
| s32 stage                 | 输入      | 算法stage         |
| 输入                      |           | 算法stage         |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

---

## PostFin

**Description:** 发送完成同步信号。

**Prototype:**

```cpp
HcclResult PostFin(Stream &stream)
```

**Parameters:**

| Parameter      | Direction | Description |
| -------------- | --------- | ----------- |
| Stream &stream | 输入      | Stream对象  |
| 输入           |           | Stream对象  |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

---

## PostFinAck

**Description:** 发送完成应答同步信号。

**Prototype:**

```cpp
HcclResult PostFinAck(Stream &stream)
```

**Parameters:**

| Parameter      | Direction | Description |
| -------------- | --------- | ----------- |
| Stream &stream | 输入      | Stream对象  |
| 输入           |           | Stream对象  |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

---

## PostReady

**Description:** 发送ready同步信号。

**Prototype:**

```cpp
HcclResult PostReady(Stream &stream)
```

**Parameters:**

| Parameter      | Direction | Description |
| -------------- | --------- | ----------- |
| Stream &stream | 输入      | Stream对象  |
| 输入           |           | Stream对象  |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

---

## PushTaskLogicInfo

**Description:** 保存一个逻辑task信息。

**Prototype:**

```cpp
void PushTaskLogicInfo(TaskLogicInfo &taskLogicInfo)
```

**Parameters:**

| Parameter                    | Direction | Description |
| ---------------------------- | --------- | ----------- |
| TaskLogicInfo &taskLogicInfo | 输入      | Task信息    |
| 输入                         |           | Task信息    |

**Return Value:** 无。

---

## Read

**Description:** 单边读数据。

**Prototype:**

```cpp
HcclResult Read(const void *localAddr, UserMemType remoteMemType, u64 remoteOffset, u64 len, Stream &stream)
```

**Parameters:**

| Parameter                 | Direction | Description      |
| ------------------------- | --------- | ---------------- |
| const void *localAddr     | 输入      | 算法step信息     |
| 输入                      |           | 算法step信息     |
| UserMemType remoteMemType | 输入      | 远端用户内存类型 |
| 输入                      |           | 远端用户内存类型 |
| u64 remoteOffset          | 输入      | 远端地址偏移     |
| 输入                      |           | 远端地址偏移     |
| u64 len                   | 输入      | 数据长度         |
| 输入                      |           | 数据长度         |
| Stream &stream            | 输入      | Stream对象       |
| 输入                      |           | Stream对象       |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

---

## Recv

**Description:** TCP socket recv接口。

**Prototype:**

```cpp
HcclResult Recv(void *recvBuf, u32 recvBufLen)  // 传入接收地址和长度，接收信息
HcclResult Recv(std::string &recvMsg)  // 传入string对象接收信息
```

**Parameters:**

| Parameter            | Direction | Description      |
| -------------------- | --------- | ---------------- |
| void *recvBuf        | 输入      | 接收数据起始地址 |
| 输入                 |           | 接收数据起始地址 |
| u32 recvBufLen       | 输入      | 接收数据大小     |
| 输入                 |           | 接收数据大小     |
| std::string &recvMsg | 输入      | 接收数据         |
| 输入                 |           | 接收数据         |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

---

## RegisterInitTaskCallBack

**Description:** 注册init task callback函数。

**Prototype:**

```cpp
void RegisterInitTaskCallBack(HcclResult (*p1)(const HcclDispatcher &, hccl::Stream &))
```

**Parameters:**

| Parameter                                                | Direction | Description                |
| -------------------------------------------------------- | --------- | -------------------------- |
| HcclResult (*p1)(const HcclDispatcher &, hccl::Stream &) | 输入      | init task callback函数指针 |
| 输入                                                     |           | init task callback函数指针 |

**Return Value:** 无。

---

## RegisterLaunchTaskCallBack

**Description:** 注册launch task callback函数。

**Prototype:**

```cpp
void RegisterLaunchTaskCallBack(HcclResult (*p1)(const HcclDispatcher &, hccl::Stream &))
```

**Parameters:**

| Parameter                                                | Direction | Description                  |
| -------------------------------------------------------- | --------- | ---------------------------- |
| HcclResult (*p1)(const HcclDispatcher &, hccl::Stream &) | 输入      | launch task callback函数指针 |
| 输入                                                     |           | launch task callback函数指针 |

**Return Value:** 无。

---

## RxAck

**Description:** 本端等待对端的同步信号。

**Prototype:**

```cpp
HcclResult RxAck(Stream &stream)
```

**Parameters:**

| Parameter      | Direction | Description |
| -------------- | --------- | ----------- |
| Stream &stream | 输入      | Stream对象  |
| 输入           |           | Stream对象  |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

---

## RxAsync

**Description:** 异步接收数据，将远端指定类型地址中的数据接收到本端dst地址中。

**Prototype:**

```cpp
// 异步接收单块内存
HcclResult RxAsync(UserMemType srcMemType, u64 srcOffset, void *dst, u64 len, Stream &stream)
// 异步接收多块内存
HcclResult RxAsync(std::vector<RxMemoryInfo>& rxMems, Stream &stream)
```

**Parameters:**

| Parameter              | Direction | Description  |
| ---------------------- | --------- | ------------ |
| UserMemType srcMemType | 输入      | 算法step信息 |
| 输入                   |           | 算法step信息 |
| u64 srcOffset          | 输入      | 源地址偏移   |
| 输入                   |           | 源地址偏移   |
| void *dst              | 输入      | 目的地址     |
| 输入                   |           | 目的地址     |
| u64 len                | 输入      | 数据长度     |
| 输入                   |           | 数据长度     |
| Stream &stream         | 输入      | Stream对象   |
| 输入                   |           | Stream对象   |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

---

## RxData

**Description:** 接收数据。

**Prototype:**

```cpp
HcclResult RxData(UserMemType srcMemType, u64 srcOffset, void *dst, u64 len, Stream &stream)
```

**Parameters:**

| Parameter              | Direction | Description  |
| ---------------------- | --------- | ------------ |
| UserMemType srcMemType | 输入      | 算法step信息 |
| 输入                   |           | 算法step信息 |
| u64 srcOffset          | 输入      | 源地址偏移   |
| 输入                   |           | 源地址偏移   |
| void *dst              | 输入      | 目的地址     |
| 输入                   |           | 目的地址     |
| u64 len                | 输入      | 数据长度     |
| 输入                   |           | 数据长度     |
| Stream &stream         | 输入      | Stream对象   |
| 输入                   |           | Stream对象   |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

---

## RxDataSignal

**Description:** 本端等待对端的同步信号。

**Prototype:**

```cpp
HcclResult RxDataSignal(Stream &stream)
```

**Parameters:**

| Parameter      | Direction | Description |
| -------------- | --------- | ----------- |
| Stream &stream | 输入      | Stream对象  |
| 输入           |           | Stream对象  |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

---

## RxDone

**Description:** 接收完成。

**Prototype:**

```cpp
HcclResult RxDone(Stream &stream)
```

**Parameters:**

| Parameter      | Direction | Description |
| -------------- | --------- | ----------- |
| Stream &stream | 输入      | Stream对象  |
| 输入           |           | Stream对象  |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

---

## RxEnv

**Description:** 接收前信息准备。

**Prototype:**

```cpp
HcclResult RxEnv(Stream &stream)
```

**Parameters:**

| Parameter      | Direction | Description |
| -------------- | --------- | ----------- |
| Stream &stream | 输入      | Stream对象  |
| 输入           |           | Stream对象  |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

---

## RxPrepare

**Description:** 接收前同步准备。

**Prototype:**

```cpp
HcclResult RxPrepare(Stream &stream)
```

**Parameters:**

| Parameter      | Direction | Description |
| -------------- | --------- | ----------- |
| Stream &stream | 输入      | Stream对象  |
| 输入           |           | Stream对象  |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

---

## RxWaitDone

**Description:** 等待接收完成。

**Prototype:**

```cpp
HcclResult RxWaitDone(Stream &stream)
```

**Parameters:**

| Parameter      | Direction | Description |
| -------------- | --------- | ----------- |
| Stream &stream | 输入      | Stream对象  |
| 输入           |           | Stream对象  |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

---

## RxWithReduce

**Description:** 异步接收数据，将远端指定类型地址中的数据接收到本端dst地址中，并完成reduce操作。

**Prototype:**

```cpp
// 接收并且做reduce操作，单块内存
HcclResult RxWithReduce(UserMemType recvSrcMemType, u64 recvSrcOffset, void *recvDst, u64 recvLen, void *reduceSrc, void *reduceDst, u64 reduceDataCount, HcclDataType reduceDatatype, HcclReduceOp reduceOp, Stream &stream, const u64 reduceAttr)
// 接收并且做reduce操作，多块内存
HcclResult RxWithReduce(const std::vector<RxWithReduceMemoryInfo> &rxWithReduceMems, HcclDataType reduceDatatype, HcclReduceOp reduceOp, Stream &stream, const u64 reduceAttr)
```

**Parameters:**

| Parameter                   | Direction | Description      |
| --------------------------- | --------- | ---------------- |
| UserMemType recvSrcMemType  | 输入      | 接收用户内存类型 |
| 输入                        |           | 接收用户内存类型 |
| u64 recvSrcOffset           | 输入      | 接收源地址偏移   |
| 输入                        |           | 接收源地址偏移   |
| void *recvDst               | 输入      | 接收目的地址     |
| 输入                        |           | 接收目的地址     |
| u64 recvLen                 | 输入      | 接收长度         |
| 输入                        |           | 接收长度         |
| void *reduceSrc             | 输入      | reduce源地址     |
| 输入                        |           | reduce源地址     |
| void *reduceDst             | 输入      | reduce目的地址   |
| 输入                        |           | reduce目的地址   |
| u64 reduceDataCount         | 输入      | reduce数据量     |
| 输入                        |           | reduce数据量     |
| HcclDataType reduceDatatype | 输入      | 数据类型         |
| 输入                        |           | 数据类型         |
| HcclReduceOp reduceOp       | 输入      | Reduce类型       |
| 输入                        |           | Reduce类型       |
| Stream &stream              | 输入      | Stream对象       |
| 输入                        |           | Stream对象       |
| const u64 reduceAttr        | 输入      | Reduce属性       |
| 输入                        |           | Reduce属性       |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

---

## Send

**Description:** TCP socket send接口。

**Prototype:**

```cpp
HcclResult Send(const void *data, u64 size)  // 传入地址和发送长度，发送信息
HcclResult Send(const std::string &sendMsg)  // 传入string对象，发送信息
```

**Parameters:**

| Parameter                  | Direction | Description  |
| -------------------------- | --------- | ------------ |
| const void *data           | 输入      | 数据起始地址 |
| 输入                       |           | 数据起始地址 |
| u64 size                   | 输入      | 数据大小     |
| 输入                       |           | 数据大小     |
| const std::string &sendMsg | 输入      | 发送数据     |
| 输入                       |           | 发送数据     |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

---

## SetIfName

**Description:** 设置网卡名。

**Prototype:**

```cpp
HcclResult SetIfName(const std::string &name)
```

**Parameters:**

| Parameter               | Direction | Description |
| ----------------------- | --------- | ----------- |
| const std::string &name | 输入      | 网卡名      |
| 输入                    |           | 网卡名      |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

---

## SetIpc

**Description:** 设置ipc通信功能。

**Prototype:**

```cpp
virtual HcclResult SetIpc()
```

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

---

## SetMode

**Description:** 设置stream模式。

**Prototype:**

```cpp
HcclResult SetMode(const uint64_t stmMode)
```

**Parameters:**

| Parameter              | Direction | Description    |
| ---------------------- | --------- | -------------- |
| const uint64_t stmMode | 输入      | Stream工作模式 |
| 输入                   |           | Stream工作模式 |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

---

## SetNotifyData

**Description:** 设置notify信息。

**Prototype:**

```cpp
HcclResult SetNotifyData(HcclSignalInfo &notifyInfo)
```

**Parameters:**

| Parameter                  | Direction | Description |
| -------------------------- | --------- | ----------- |
| HcclSignalInfo &notifyInfo | 输入      | Notify信息  |
| 输入                       |           | Notify信息  |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

---

## SetReadableAddress

**Description:** 设置IP地址信息。

**Prototype:**

```cpp
HcclResult SetReadableAddress(const std::string &address)
```

**Parameters:**

| Parameter                  | Direction | Description        |
| -------------------------- | --------- | ------------------ |
| const std::string &address | 输入      | 字符串类型的Ip地址 |
| 输入                       |           | 字符串类型的Ip地址 |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

---

## SetScopeID

**Description:** 设置ScopeID。

**Prototype:**

```cpp
HcclResult SetScopeID(s32 scopeID)
```

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

---

## SetStatus

**Description:** 设置socket状态。

**Prototype:**

```cpp
void SetStatus(HcclSocketStatus status)
```

**Parameters:**

| Parameter               | Direction | Description |
| ----------------------- | --------- | ----------- |
| HcclSocketStatus status | 输入      | Socket状态  |
| 输入                    |           | Socket状态  |

**Return Value:** 无。

---

## Stream

**Description:** Stream构造函数。

**Prototype:**

```cpp
// Stream构造函数
Stream()
//Stream拷贝构造函数
Stream(const Stream &that);
//Stream移动构造函数
Stream(Stream &&that);
// 基于类型构造Stream，是stream owner
Stream(const StreamType streamType, bool isMainStream = false)
// 使用rtStream构造Stream，不是stream owner
Stream(const rtStream_t rtStream, bool isMainStream = true)
// 基于HcclComStreamInfo信息构造stream，不是stream owner
Stream(const HcclComStreamInfo &streamInfo, bool isMainStream = false)
```

**Parameters:**

| Parameter                   | Direction | Description |
| --------------------------- | --------- | ----------- |
| const StreamType streamType | 输入      | Stream类型  |
| 输入                        |           | Stream类型  |
| bool isMainStream           | 输入      | 是否是主流  |
| 输入                        |           | 是否是主流  |

**Return Value:** 无。

---

## \~Stream

**Description:** Stream析构函数。

**Prototype:**

```cpp
~Stream()
```

**Return Value:** 无。

---

## TagMachinePara

**Description:** TagMachinePara构造函数。

**Prototype:**

```cpp
TagMachinePara()  // 默认构造函数
TagMachinePara(const struct TagMachinePara &that)  // 拷贝构造函数
```

**Parameters:**

| Parameter                         | Direction | Description          |
| --------------------------------- | --------- | -------------------- |
| const struct TagMachinePara &that | 输入      | TagMachinePara结构体 |
| 输入                              |           | TagMachinePara结构体 |

**Return Value:** 无。

---

## TransDataDef

**Description:** TransDataDef构造函数。

**Prototype:**

```cpp
TransDataDef()   // 默认构造函数
TransDataDef(u64 srcBuf, u64 dstBuf, u64 count, HcclDataType dataType, bool errorFlag = false, u32 tableId = DEFAULT_TABLE_ID_VALUE, s64 globalStep = DEFAULT_GLOBAL_STEP_VALUE)  //构造函数
```

**Parameters:**

| Parameter             | Direction | Description |
| --------------------- | --------- | ----------- |
| u64 srcBuf            | 输入      | 源地址      |
| 输入                  |           | 源地址      |
| u64 dstBuf            | 输入      | 目的地址    |
| 输入                  |           | 目的地址    |
| u64 count             | 输入      | 数据量      |
| 输入                  |           | 数据量      |
| HcclDataType dataType | 输入      | 数据类型    |
| 输入                  |           | 数据类型    |
| bool errorFlag        | 输入      | Error标记   |
| 输入                  |           | Error标记   |
| u32 tableId           | 输入      | Table id    |
| 输入                  |           | Table id    |
| s64 globalStep        | 输入      | 全局step    |
| 输入                  |           | 全局step    |

**Return Value:** 无。

---

## Transport

**Description:** Transport构造函数。

**Prototype:**

```cpp
Transport(TransportBase *pimpl)
Transport(TransportType type, TransportPara& para, const HcclDispatcher dispatcher, const std::unique_ptr<NotifyPool> &notifyPool, MachinePara &machinePara,  const TransportDeviceP2pData &transDevP2pData = TransportDeviceP2pData())
```

**Parameters:**

| Parameter                                         | Direction | Description       |
| ------------------------------------------------- | --------- | ----------------- |
| TransportBase *pimpl                              | 输入      | TransportBase指针 |
| 输入                                              |           | TransportBase指针 |
| TransportType type                                | 输入      | Transport类型     |
| 输入                                              |           | Transport类型     |
| TransportPara& para                               | 输入      | Transport参数     |
| 输入                                              |           | Transport参数     |
| const HcclDispatcher dispatcher                   | 输入      | Dispatcher handle |
| 输入                                              |           | Dispatcher handle |
| const std::unique_ptr`<NotifyPool>` &notifyPool | 输入      | Notify pool指针   |
| 输入                                              |           | Notify pool指针   |
| MachinePara &machinePara                          | 输入      | 建链相关参数      |
| 输入                                              |           | 建链相关参数      |
| const TransportDeviceP2pData &transDevP2pData     | 输入      | device侧相关数据  |
| 输入                                              |           | device侧相关数据  |

**Return Value:** 无。

---

## \~Transport

**Description:** Transport析构函数。

**Prototype:**

```cpp
~Transport()
```

**Return Value:** 无。

---

## TransportDeviceP2pData

**Description:** TransportDeviceP2pData构造函数。

**Prototype:**

```cpp
TransportDeviceP2pData () //默认构造函数
TransportDeviceP2pData (const struct TransportDeviceP2pData&that) //拷贝构造函数
TransportDeviceP2pData(void *inputBufferPtr,void *outputBufferPtr,std::shared_ptr<LocalIpcNotify> ipcPreWaitNotify,std::shared_ptr<LocalIpcNotify> ipcPostWaitNotify,
std::shared_ptr<RemoteNotify> ipcPreRecordNotify,std::shared_ptr<RemoteNotify> ipcPostRecordNotify,LinkType linkType) //构造函数
```

**Parameters:**

| Parameter                                             | Direction | Description                  |
| ----------------------------------------------------- | --------- | ---------------------------- |
| const struct TransportDeviceP2pData&that              | 输入      | TransportDeviceP2pData结构体 |
| 输入                                                  |           | TransportDeviceP2pData结构体 |
| void *inputBufferPtr                                  | 输入      | Receive Buffer指针           |
| 输入                                                  |           | Receive Buffer指针           |
| void *outputBufferPtr                                 | 输入      | Send Buffer指针              |
| 输入                                                  |           | Send Buffer指针              |
| std::shared_ptr`<LocalIpcNotify>` ipcPreWaitNotify  | 输入      | 本地IPC Notify指针           |
| 输入                                                  |           | 本地IPC Notify指针           |
| std::shared_ptr`<LocalIpcNotify>` ipcPostWaitNotify | 输入      | 本地IPC Notify指针           |
| 输入                                                  |           | 本地IPC Notify指针           |
| std::shared_ptr`<RemoteNotify>` ipcPreRecordNotify  | 输入      | 本地IPC Notify指针           |
| 输入                                                  |           | 本地IPC Notify指针           |
| std::shared_ptr`<RemoteNotify>` ipcPostRecordNotify | 输入      | 本地IPC Notify指针           |
| 输入                                                  |           | 本地IPC Notify指针           |
| LinkType linkType                                     | 输入      | 链路类型(ONCHIP/PCIE/ROCE…) |
| 输入                                                  |           | 链路类型(ONCHIP/PCIE/ROCE…) |

**Return Value:** 无。

---

## TxAck

**Description:** 本端发送同步信号到对端。

**Prototype:**

```cpp
HcclResult TxAck(Stream &stream)
```

**Parameters:**

| Parameter      | Direction | Description |
| -------------- | --------- | ----------- |
| Stream &stream | 输入      | Stream对象  |
| 输入           |           | Stream对象  |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

---

## TxAsync

**Description:** 异步发送数据，将本端src地址的数据发送到远端指定类型地址中。

**Prototype:**

```cpp
// 单块内存TxAsync
HcclResult TxAsync(UserMemType dstMemType, u64 dstOffset, const void *src, u64 len, Stream &stream)
// 多块内存TxAsync
HcclResult TxAsync(std::vector<TxMemoryInfo>& txMems, Stream &stream)
```

**Parameters:**

| Parameter                             | Direction | Description      |
| ------------------------------------- | --------- | ---------------- |
| UserMemType dstMemType                | 输入      | 对端用户内存类型 |
| 输入                                  |           | 对端用户内存类型 |
| u64 dstOffset                         | 输入      | 对端内存偏移     |
| 输入                                  |           | 对端内存偏移     |
| const void *src                       | 输入      | 源地址           |
| 输入                                  |           | 源地址           |
| u64 len                               | 输入      | 发送数据大小     |
| 输入                                  |           | 发送数据大小     |
| Stream &stream                        | 输入      | Stream对象       |
| 输入                                  |           | Stream对象       |
| std::vector`<TxMemoryInfo>`& txMems | 输入      | 发送内存信息     |
| 输入                                  |           | 发送内存信息     |
| Stream &stream                        | 输入      | Stream对象       |
| 输入                                  |           | Stream对象       |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

---

## TxData

**Description:** 发送数据。

**Prototype:**

```cpp
HcclResult TxData(UserMemType dstMemType, u64 dstOffset, const void *src, u64 len, Stream &stream)
```

**Parameters:**

| Parameter              | Direction | Description  |
| ---------------------- | --------- | ------------ |
| UserMemType dstMemType | 输入      | 算法step信息 |
| 输入                   |           | 算法step信息 |
| u64 dstOffset          | 输入      | 目的偏移     |
| 输入                   |           | 目的偏移     |
| const void *src        | 输入      | 源地址       |
| 输入                   |           | 源地址       |
| u64 len                | 输入      | 数据长度     |
| 输入                   |           | 数据长度     |
| Stream &stream         | 输入      | Stream对象   |
| 输入                   |           | Stream对象   |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

---

## TxDataSignal

**Description:** 本端发送同步信号到对端。

**Prototype:**

```cpp
HcclResult TxDataSignal(Stream &stream)
```

**Parameters:**

| Parameter      | Direction | Description |
| -------------- | --------- | ----------- |
| Stream &stream | 输入      | Stream对象  |
| 输入           |           | Stream对象  |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

---

## TxDone

**Description:** 发送完成。

**Prototype:**

```cpp
HcclResult TxDone(Stream &stream)
```

**Parameters:**

| Parameter      | Direction | Description |
| -------------- | --------- | ----------- |
| Stream &stream | 输入      | Stream对象  |
| 输入           |           | Stream对象  |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

---

## TxEnv

**Description:** 发送前信息准备。

**Prototype:**

```cpp
HcclResult TxEnv(const void *ptr, const u64 len, Stream &stream)
```

**Parameters:**

| Parameter       | Direction | Description  |
| --------------- | --------- | ------------ |
| const void *ptr | 输入      | 算法step信息 |
| 输入            |           | 算法step信息 |
| const u64 len   | 输入      | 数据长度     |
| 输入            |           | 数据长度     |
| Stream &stream  | 输入      | Stream对象   |
| 输入            |           | Stream对象   |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

---

## TxPrepare

**Description:** 发送前同步准备。

**Prototype:**

```cpp
HcclResult TxPrepare(Stream &stream)
```

**Parameters:**

| Parameter      | Direction | Description |
| -------------- | --------- | ----------- |
| Stream &stream | 输入      | Stream对象  |
| 输入           |           | Stream对象  |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

---

## TxWaitDone

**Description:** 等待发送完成。

**Prototype:**

```cpp
HcclResult TxWaitDone(Stream &stream)
```

**Parameters:**

| Parameter      | Direction | Description |
| -------------- | --------- | ----------- |
| Stream &stream | 输入      | Stream对象  |
| 输入           |           | Stream对象  |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

---

## TxWithReduce

**Description:** 异步发送数据，将本端src地址的数据发送到远端指定类型地址中，并完成reduce操作。

**Prototype:**

```cpp
// 发送一块内存数据
HcclResult TxWithReduce(UserMemType dstMemType, u64 dstOffset, const void *src, u64 len, const HcclDataType datatype, HcclReduceOp redOp, Stream &stream)
// 发送多块内存数据
HcclResult TxWithReduce(const std::vector<TxMemoryInfo> &txWithReduceMems, const HcclDataType datatype, HcclReduceOp redOp, Stream &stream)
```

**Parameters:**

| Parameter                   | Direction | Description      |
| --------------------------- | --------- | ---------------- |
| UserMemType dstMemType      | 输入      | 对端用户内存类型 |
| 输入                        |           | 对端用户内存类型 |
| u64 dstOffset               | 输入      | 对端内存偏移     |
| 输入                        |           | 对端内存偏移     |
| const void *src             | 输入      | 源地址           |
| 输入                        |           | 源地址           |
| u64 len                     | 输入      | 发送数据大小     |
| 输入                        |           | 发送数据大小     |
| const HcclDataType datatype | 输入      | 数据类型         |
| 输入                        |           | 数据类型         |
| HcclReduceOp redOp          | 输入      | Reduce类型       |
| 输入                        |           | Reduce类型       |
| Stream &stream              | 输入      | Stream对象       |
| 输入                        |           | Stream对象       |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

---

## Wait

**Description:** Notify wait任务。

**Prototype:**

```cpp
HcclResult Wait(Stream& stream, HcclDispatcher dispatcher, s32 stage = INVALID_VALUE_STAGE, u32 timeOut = NOTIFY_DEFAULT_WAIT_TIME)
```

**Parameters:**

| Parameter                 | Direction | Description       |
| ------------------------- | --------- | ----------------- |
| Stream& stream            | 输入      | Stream对象        |
| 输入                      |           | Stream对象        |
| HcclDispatcher dispatcher | 输入      | Dispatcher handle |
| 输入                      |           | Dispatcher handle |
| s32 stage                 | 输入      | 算法stage         |
| 输入                      |           | 算法stage         |
| u32 timeOut               | 输入      | Notify超时时间    |
| 输入                      |           | Notify超时时间    |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

---

## WaitFin

**Description:** 接收完成同步信号。

**Prototype:**

```cpp
HcclResult WaitFin(Stream &stream)
```

**Parameters:**

| Parameter      | Direction | Description |
| -------------- | --------- | ----------- |
| Stream &stream | 输入      | Stream对象  |
| 输入           |           | Stream对象  |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

---

## WaitFinAck

**Description:** 接收完成应答同步信号。

**Prototype:**

```cpp
HcclResult WaitFinAck(Stream &stream)
```

**Parameters:**

| Parameter      | Direction | Description |
| -------------- | --------- | ----------- |
| Stream &stream | 输入      | Stream对象  |
| 输入           |           | Stream对象  |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

---

## WaitReady

**Description:** 等待ready同步信号。

**Prototype:**

```cpp
HcclResult WaitReady(Stream &stream)
```

**Parameters:**

| Parameter      | Direction | Description |
| -------------- | --------- | ----------- |
| Stream &stream | 输入      | Stream对象  |
| 输入           |           | Stream对象  |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

---

## Write

**Description:** 单边写数据。

**Prototype:**

```cpp
HcclResult Write(const void *localAddr, UserMemType remoteMemType, u64 remoteOffset, u64 len, Stream &stream)
```

**Parameters:**

| Parameter                 | Direction | Description      |
| ------------------------- | --------- | ---------------- |
| const void *localAddr     | 输入      | 算法step信息     |
| 输入                      |           | 算法step信息     |
| UserMemType remoteMemType | 输入      | 远端用户内存类型 |
| 输入                      |           | 远端用户内存类型 |
| u64 remoteOffset          | 输入      | 远端地址偏移     |
| 输入                      |           | 远端地址偏移     |
| u64 len                   | 输入      | 数据长度         |
| 输入                      |           | 数据长度         |
| Stream &stream            | 输入      | Stream对象       |
| 输入                      |           | Stream对象       |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

---

## alloc

**Description:** 根据输入去申请device内存。

**Prototype:**

```cpp
static HostMem alloc(u64 size, bool isRtsMem = true)
```

**Parameters:**

| Parameter     | Direction | Description     |
| ------------- | --------- | --------------- |
| u64 size,     | 输入      | 内存大小        |
| 输入          |           | 内存大小        |
| bool isRtsMem | 输入      | 是否通过rts申请 |
| 输入          |           | 是否通过rts申请 |

**Return Value:** Host Mem对象。

---

## alloc

**Description:** 根据输入去申请device内存。

**Prototype:**

```cpp
static DeviceMem alloc(u64 size, bool level2Address = false)
```

**Parameters:**

| Parameter          | Direction | Description    |
| ------------------ | --------- | -------------- |
| u64 size           | 输入      | 内存大小       |
| 输入               |           | 内存大小       |
| bool level2Address | 输入      | 是否是二级地址 |
| 输入               |           | 是否是二级地址 |

**Return Value:** DeviceMem对象。

---

## cqId

**Description:** 获取sq id。

**Prototype:**

```cpp
u32 cqId() const
```

**Return Value:** Cq id。

---

## create

**Description:** 用输入地址和大小构造HostMem对象，不会去申请、释放host内存。

**Prototype:**

```cpp
static HostMem create(void *ptr, u64 size)
```

**Parameters:**

| Parameter | Direction | Description |
| --------- | --------- | ----------- |
| void *ptr | 输入      | 内存地址    |
| 输入      |           | 内存地址    |
| u64 size  | 输入      | 内存大小    |
| 输入      |           | 内存大小    |

**Return Value:** Host Mem对象。

---

## create

**Description:** 用输入地址和大小构造DeviceMem对象，不会去申请、释放device内存。

**Prototype:**

```cpp
static DeviceMem create(void *ptr, u64 size)
```

**Parameters:**

| Parameter | Direction | Description |
| --------- | --------- | ----------- |
| void *ptr | 输入      | 内存地址    |
| 输入      |           | 内存地址    |
| u64 size  | 输入      | 内存大小    |
| 输入      |           | 内存大小    |

**Return Value:** DeviceMem对象。

---

## id

**Description:** 获取stream id。

**Prototype:**

```cpp
s32 id() const
```

**Return Value:** Stream id。

---

## ptr

**Description:** 获取host mem地址。

**Prototype:**

```cpp
void *ptr() const
```

**Return Value:** host mem地址。

---

## ptr

**Description:** 获取stream ptr。

**Prototype:**

```cpp
void *ptr() const
```

**Return Value:** Stream ptr。

---

## ptr

**Description:** 获取device mem地址

**Prototype:**

```cpp
void *ptr() const
```

**Return Value:** device mem地址。

---

## range

**Description:** 在当前mem实例中截取一段形成新的Mem实例。

**Prototype:**

```cpp
HostMem range(u64 offset, u64 size) const
```

**Parameters:**

| Parameter  | Direction | Description |
| ---------- | --------- | ----------- |
| u64 offset | 输入      | 偏移大小    |
| 输入       |           | 偏移大小    |
| u64 size   | 输入      | 新实例大小  |
| 输入       |           | 新实例大小  |

**Return Value:** Host Mem对象。

---

## range

**Description:** 在当前mem实例中截取一段形成新的Mem实例。

**Prototype:**

```cpp
DeviceMem range(u64 offset, u64 size) const
```

**Parameters:**

| Parameter  | Direction | Description |
| ---------- | --------- | ----------- |
| u64 offset | 输入      | 偏移大小    |
| 输入       |           | 偏移大小    |
| u64 size   | 输入      | 新实例大小  |
| 输入       |           | 新实例大小  |

**Return Value:** Device Mem对象。

---

## size

**Description:** 获取host mem大小。

**Prototype:**

```cpp
u64 size() const
```

**Return Value:** host mem大小。

---

## size

**Description:** 获取device mem大小。

**Prototype:**

```cpp
u64 size() const
```

**Return Value:** device mem大小。

---

## sqId

**Description:** 获取sq id。

**Prototype:**

```cpp
u32 sqId() const
```

**Return Value:** Sq id。

---

## static Post

**Description:** Notify post任务。

**Prototype:**

```cpp
static HcclResult Post(Stream& stream, HcclDispatcher dispatcherPtr, const std::shared_ptr<LocalNotify> &notify, s32 stage = INVALID_VALUE_STAGE)
```

**Parameters:**

| Parameter                                      | Direction | Description       |
| ---------------------------------------------- | --------- | ----------------- |
| Stream& stream                                 | 输入      | Stream对象        |
| 输入                                           |           | Stream对象        |
| HcclDispatcher dispatcher                      | 输入      | Dispatcher handle |
| 输入                                           |           | Dispatcher handle |
| const std::shared_ptr`<LocalNotify>` &notify | 输入      | Notify对象指针    |
| 输入                                           |           | Notify对象指针    |
| s32 stage                                      | 输入      | 算法stage         |
| 输入                                           |           | 算法stage         |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

---

## static Wait

**Description:** Notify wait任务。

**Prototype:**

```cpp
static HcclResult Wait(Stream& stream, HcclDispatcher dispatcherPtr, const std::shared_ptr<LocalNotify> &notify, s32 stage = INVALID_VALUE_STAGE, u32 timeOut = NOTIFY_DEFAULT_WAIT_TIME)
```

**Parameters:**

| Parameter                                      | Direction | Description       |
| ---------------------------------------------- | --------- | ----------------- |
| Stream& stream                                 | 输入      | Stream对象        |
| 输入                                           |           | Stream对象        |
| HcclDispatcher dispatcher                      | 输入      | Dispatcher handle |
| 输入                                           |           | Dispatcher handle |
| const std::shared_ptr`<LocalNotify>` &notify | 输入      | Notify对象指针    |
| 输入                                           |           | Notify对象指针    |
| s32 stage                                      | 输入      | 算法stage         |
| 输入                                           |           | 算法stage         |
| u32 timeOut                                    | 输入      | Notify超时时间    |
| 输入                                           |           | Notify超时时间    |

**Return Value:** HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

---
