#include "coll_all_gather_new_executor.h"

namespace hccl {
CollAllGatherNewExecutor::CollAllGatherNewExecutor(const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollAllGatherExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = false;
}

//流数量计算
HcclResult CollAllGatherNewExecutor::CalcStreamNum(u32& streamNum)
{
    /*单ring流数量计算
    streamNum = 0;  // 只使用主stream，不需要额外的子stream
    HCCL_INFO("[CollAllGatherNewExecutor][CalcStreamNum] tag[%s] streamNum[%u]",
        tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
    */

}

HcclResult CollAllGatherNewExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    // 实现通信信息计算逻辑
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel1CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherNewExecutor::CalcLevel0CommInfo(TransportMemType inputType, TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
   // 实现Level0通信信息计算逻辑 下为mesh逻辑拓扑计算样例
   /*
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_MESH);
    commParaLevel0.meshSinglePlane = (topoAttr_.deviceType == DevType::DEV_TYPE_910B) &&
        !topoMatcher_->GetExternalInputHcclDeterministic() && (workflowMode_ != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    return HCCL_SUCCESS;
    */

}

HcclResult CollAllGatherNewExecutor::CalcTransportMemType(TransportMemType &inputType, TransportMemType &outputType)
{
    // 实现传输内存类型计算逻辑
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        inputType = TransportMemType::CCL_INPUT;
        outputType = TransportMemType::CCL_OUTPUT;
    } else {
        inputType = TransportMemType::PARAM_INPUT;
        outputType = TransportMemType::PARAM_OUTPUT;
    }
    HCCL_INFO("[CollAllGatherNewExecutor][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d]",
        tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

//单环ring算法
HcclResult CollAllGatherNewExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{

    //编写算法逻辑  具体task的执行可以通过Runtemplate调用自定义的模板来实现

    
    return HCCL_SUCCESS;
}


REGISTER_EXEC("AllGatherNewExecutor", AllGatherNew, CollAllGatherNewExecutor);

} // namespace hccl

//可以参照all_gather_ring_execuotor.cc、all_gather_mesh_executor.cc等文件的实现方式，