/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALLGATHER_MESH_H
#define ALLGATHER_MESH_H

#include "coll_allgather_executor.h"

namespace hccl {
class AllgatherMesh : public CollAllgatherExecutor {
public:
    explicit AllgatherMesh(const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher> &topoMatcher);
    ~AllgatherMesh() = default;

private:
    /* *************** Resource calculation *************** */
    std::set<u32> commTargetUserRankSet_;
    bool isZeroCopy_ = false;
    HcclResult CalcStreamNum(u32& streamNum) override;
    HcclResult CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcLevel0CommInfo(TransportMemType inputType, TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcLevel1CommInfo(TransportMemType inputType, TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport);
    HcclResult CalcLevel2CommInfo(TransportMemType inputType, TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcTransportMemType(TransportMemType &inputType, TransportMemType &outputType);
    void ParseParam(const OpParam& param) override;
    
    /* *************** Algorithm orchestration *************** */
    u64 CalcLoopMaxCount(const u64 cclBuffSize, const u32 unitSize) override;
    bool IsDataSplitForRdmaSdmaConcurrent(const u64 curSize) override;
    HcclResult KernelRun(const OpParam &param, ExecMem &execMem) override;
    HcclResult KernelRunInterServer(const OpParam &param, ExecMem &execMem) override;
    HcclResult KernelRunIntraServer(const OpParam &param, ExecMem &execMem) override;
    HcclResult Orchestrate(const OpParam& param, AlgResourceResponse& algRes) override;
    bool IsHugeData(const u64 curSize);
    bool IsSmallData(const u64 size);
    bool IsDataSplitForRdmaSdmaConcurrent(const u64 curSize);
    HcclResult PrepareAllgatherSlice(u32 sliceNum, u64 inputMemSize,
        std::vector<Slice> &dataSegsSlice) const;
};

} // namespace hccl

#endif /* ALLGATHER_MESH_H */