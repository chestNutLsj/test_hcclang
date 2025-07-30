#ifndef COLL_ALLGATHER_NEW_EXECUTOR_H
#define COLL_ALLGATHER_NEW_EXECUTOR_H
#include "coll_all_gather_executor.h"

namespace hccl {
class CollAllGatherNewExecutor : public CollAllGatherExecutor {
public:
    explicit CollAllGatherNewExecutor(const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher> &topoMatcher);
    ~CollAllGatherNewExecutor() = default;

private:
    /* *************** 资源计算 *************** */
    HcclResult CalcStreamNum(u32& streamNum) override; //override 表示该函数是对基类中同名虚函数的重写，确保多态调用时行为正确。
    HcclResult CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcLevel0CommInfo(TransportMemType inputType, TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcTransportMemType(TransportMemType &inputType, TransportMemType &outputType) override;
    
    /* *************** 算法编排 *************** */
    u64 CalcLoopMaxCount(const u64 cclBuffSize, const u32 unitSize) override;
    HcclResult KernelRun(const OpParam &param, ExecMem &execMem) override;
    HcclResult SingleRingAllGather(const std::string& tag,DeviceMem inputMem,DeviceMem outputMem,const u64 count,const HcclDataType dataType,
            std::vector<std::vector<Slice>> multRingsSliceZero,Stream stream,s32 profStage,const u64 baseOffset,const HcomCollOpInfo* opInfo) override;
};

} // namespace hccl
#endif