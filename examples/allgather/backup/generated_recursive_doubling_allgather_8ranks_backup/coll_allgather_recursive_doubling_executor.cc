/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_allgather_recursive_doubling_executor.h"

namespace hccl {
AllgatherRecursiveDoubling::AllgatherRecursiveDoubling(const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollAllgatherExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = false;
}

HcclResult AllgatherRecursiveDoubling::CalcStreamNum(u32& streamNum)
{
    u32 totalStreamNum = (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING ? LEVEL0_PLANE_NUM_IN_NPRING_DOUBLE :
        LEVEL0_PLANE_NUM_IN_NPRING_SINGLE);
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        totalStreamNum *= STREAM_NUM_FOR_DMAREDUCE_ONE_RING;
    }
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB &&
        GetExternalInputEnableRdmaSdmaConcurrent()) {
        totalStreamNum += (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) ? LEVEL0_PLANE_NUM_IN_NPRING_DOUBLE :
        LEVEL0_PLANE_NUM_IN_NPRING_SINGLE;
    }
    streamNum = totalStreamNum - 1;
    HCCL_INFO("[AllgatherRecursiveDoubling][CalcStreamNum] tag[%s] streamNum_[%u]",
        tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult AllgatherRecursiveDoubling::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel1CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult AllgatherRecursiveDoubling::CalcLevel0CommInfo(TransportMemType inputType, TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    // RecursiveDoubling topology communication setup
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_RING_INNER);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult AllgatherRecursiveDoubling::CalcLevel1CommInfo(TransportMemType inputType, TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    bool isSelectAHC = (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC ||
        algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE);
    CommPlane commPlaneLevel1 = isSelectAHC ? COMM_LEVEL1_AHC : COMM_LEVEL1;
    CHK_RET(CheckCommSize(commPlaneLevel1, COMM_INDEX_0 + 1));
    SubCommInfo level1CommInfo = GetSubCommInfo(commPlaneLevel1, COMM_INDEX_0);

    CommParaInfo commParaLevel1(COMM_LEVEL1, CommType::COMM_TAG_RING_INNER);
    if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
        commParaLevel1.commType = CommType::COMM_TAG_NONUNIFORM_BRUCK;
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
        commParaLevel1.commType = CommType::COMM_TAG_NONUNIFORM_HIERARCHICAL_RING;
    }
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel1, opTransport[COMM_LEVEL1], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult AllgatherRecursiveDoubling::CalcLevel2CommInfo(TransportMemType inputType, TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    if( algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC ||
        algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE) {
        HCCL_INFO("[AllgatherRecursiveDoubling][CalcLevel2CommInfo] select AHC bypass level2 comm calculate");
        return HCCL_SUCCESS;
    }

    CommParaInfo commParaLevel2(COMM_LEVEL2, CommType::COMM_TAG_MAX);
    if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NHR) {
        commParaLevel2.commType = CommType::COMM_TAG_NONUNIFORM_HIERARCHICAL_RING;
        HCCL_INFO("[%s]Calc NHRCommInfo", __func__);
    } else if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NB) {
        commParaLevel2.commType = CommType::COMM_TAG_NONUNIFORM_BRUCK;
        HCCL_INFO("[%s]Calc NBCommInfo", __func__);
    } else {
        commParaLevel2.commType = CommType::COMM_TAG_RING_INNER;
        HCCL_INFO("[%s]Calc RingCommInfo", __func__);
    }
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel2, opTransport[COMM_LEVEL2], inputType, outputType));
    return HCCL_SUCCESS;
}

void AllgatherRecursiveDoubling::ParseParam(const OpParam& param)
{
    tag_ = param.tag;
    root_ = param.root;
    aicpuUnfoldMode_ = param.aicpuUnfoldMode;
    opType_ = param.opType;
    isZeroCopy_ = param.isZeroCopy;
}

HcclResult AllgatherRecursiveDoubling::CalcTransportMemType(TransportMemType &inputType, TransportMemType &outputType)
{
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        inputType = TransportMemType::CCL_INPUT;
        outputType = TransportMemType::CCL_OUTPUT;
    } else {
        inputType = TransportMemType::PARAM_INPUT;
        outputType = TransportMemType::PARAM_OUTPUT;
    }
    HCCL_INFO("[AllgatherRecursiveDoubling][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d]",
        tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

u64 AllgatherRecursiveDoubling::CalcLoopMaxCount(const u64 cclBuffSize, const u32 unitSize)
{
    u64 maxCountPerLoop = cclBuffSize / (topoAttr_.userRankSize * unitSize);
    return maxCountPerLoop;
}

bool AllgatherRecursiveDoubling::IsDataSplitForRdmaSdmaConcurrent(const u64 curSize)
{
    bool isLargeSize = (curSize >= HCCL_SPLIT_SIZE_INTER_SERVER);
    return GetExternalInputEnableRdmaSdmaConcurrent() && (topoAttr_.serverNum > 1) && isLargeSize;
}

HcclResult AllgatherRecursiveDoubling::KernelRun(const OpParam &param, ExecMem &execMem)
{
    u32 perDataSize = 0;
    CHK_RET(SalGetDataTypeSize(param.DataDes.dataType, perDataSize));
    CHK_PRT_RET(perDataSize == 0,
        HCCL_ERROR("[AllgatherRecursiveDoubling][KernelRun]errNo[0x%016llx] datatype[%d] is invalid",
            HCCL_ERROR_CODE(HCCL_E_PARA), param.DataDes.dataType), HCCL_E_PARA);

    // Get communication info for level0 (intra-server)
    CHK_RET(CheckCommSize(COMM_LEVEL0, 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    u32 localRank = level0CommInfo.localRank;
    u32 localRankSize = level0CommInfo.localRankSize;

    // Template will handle input-to-output data copy to avoid duplication
    u64 inputMemSize = execMem.inputMem.size();
    // Prepare slice information for algorithm
    std::vector<Slice> dataSlices;
    u64 sliceSize = inputMemSize;
    for (u32 i = 0; i < localRankSize; i++) {
        Slice slice;
        slice.offset = i * sliceSize;
        slice.size = sliceSize;
        dataSlices.push_back(slice);
    }

    // Create and run the algorithm template
    std::unique_ptr<AlgTemplateBase> algorithmTemplate = 
        AlgTemplateRegistry::Instance().GetAlgTemplate(
            TemplateType::TEMPLATE_ALLGATHER_RECURSIVE_DOUBLING, dispatcher_);
    CHK_SMART_PTR_NULL(algorithmTemplate);

    // Prepare algorithm with parameters
    CHK_RET(algorithmTemplate->Prepare(execMem.outputMem, execMem.outputMem, execMem.inputMem, 
        execMem.count, param.DataDes.dataType, param.stream, 
        HCCL_REDUCE_RESERVED, INVALID_VALUE_RANKID, dataSlices, 0));

    // Register profiler
    CHK_RET(algorithmTemplate->RegisterProfiler(
        (localRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + localRank,
        PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, param.stream));

    // Execute the algorithm
    CHK_RET(RunTemplate(algorithmTemplate, level0CommInfo));

    HCCL_INFO("[AllgatherRecursiveDoubling][KernelRun] Algorithm execution completed.");
    return HCCL_SUCCESS;
}

HcclResult AllgatherRecursiveDoubling::KernelRunInterServer(const OpParam &param, ExecMem &execMem)
{
    HCCL_INFO("[AllgatherRecursiveDoubling][KernelRunInterServer] Delegating to KernelRun");
    return KernelRun(param, execMem);
    return HCCL_SUCCESS;
}

HcclResult AllgatherRecursiveDoubling::KernelRunIntraServer(const OpParam &param, ExecMem &execMem)
{
    HCCL_INFO("[AllgatherRecursiveDoubling][KernelRunIntraServer] Delegating to KernelRun");
    return KernelRun(param, execMem);
    return HCCL_SUCCESS;
}

HcclResult AllgatherRecursiveDoubling::Orchestrate(const OpParam& param, AlgResourceResponse& algRes)
{
    HCCL_INFO("[AllgatherRecursiveDoubling][Orchestrate] Starting algorithm orchestration");

    // Parse operation parameters
    ParseParam(param);

    // Calculate communication info and resource requirements
    CHK_RET(CalcCommInfo(algRes.opTransportResponse));

    // Algorithm orchestration - prepare execution memory
    ExecMem execMem;
    execMem.count = param.DataDes.count;
    execMem.inputMem = algRes.cclInputMem;
    execMem.outputMem = algRes.cclOutputMem;
    execMem.inputPtr = param.inputPtr;
    execMem.outputPtr = param.outputPtr;

    // Execute main algorithm through KernelRun
    CHK_RET(KernelRun(param, execMem));
    return HCCL_SUCCESS;
}

REGISTER_EXEC("AllgatherRecursiveDoublingExecutor", AllgatherRecursiveDoubling, AllgatherRecursiveDoubling);

} // namespace hccl