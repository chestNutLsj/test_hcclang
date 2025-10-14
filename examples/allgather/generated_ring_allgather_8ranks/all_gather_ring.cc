// SPDX-License-Identifier: GPL-2.0-only

/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "all_gather_ring.h"
#include "alg_template_register.h"

namespace hccl {
AllGatherRing::AllGatherRing(const HcclDispatcher dispatcher) : AlgTemplateBase(dispatcher)
{
}

AllGatherRing::~AllGatherRing()
{
}

// Communication primitives
HcclResult AllGatherRing::TxVector(const LINK &link, const std::vector<Slice> &txSlices)
{
    std::vector<TxMemoryInfo> txMems;
    for (const Slice &txSlice : txSlices) {
        DeviceMem srcMem = outputMem_.range(txSlice.offset, txSlice.size);
        HCCL_DEBUG("tx srcMem[%p] range[%llu] size[%llu] ", srcMem.ptr(), txSlice.offset, txSlice.size);
        txMems.emplace_back(TxMemoryInfo{UserMemType::OUTPUT_MEM, txSlice.offset + baseOffset_,
            srcMem.ptr(), txSlice.size});
    }
    CHK_RET(link->TxAsync(txMems, stream_));
    return HCCL_SUCCESS;
}

HcclResult AllGatherRing::RxVector(const LINK &link, const std::vector<Slice> &rxSlices)
{
    std::vector<RxMemoryInfo> rxMems;
    for (const Slice &rxSlice : rxSlices) {
        DeviceMem dstMem = outputMem_.range(rxSlice.offset, rxSlice.size);
        HCCL_DEBUG("rx dstMem[%p] range[%llu], size[%llu] ",  dstMem.ptr(),
            rxSlice.offset, rxSlice.size);
        rxMems.emplace_back(RxMemoryInfo{UserMemType::OUTPUT_MEM, rxSlice.offset + baseOffset_,
            dstMem.ptr(), rxSlice.size});
    }
    CHK_RET(link->RxAsync(rxMems, stream_));
    return HCCL_SUCCESS;
}

HcclResult AllGatherRing::Tx(const LINK &link, const Slice &txSlice)
{
    DeviceMem srcMem = outputMem_.range(txSlice.offset, txSlice.size);
    HCCL_DEBUG("tx srcMem[%p] range[%llu] size[%llu] ", srcMem.ptr(), txSlice.offset, txSlice.size);
    CHK_RET(link->TxAsync(UserMemType::OUTPUT_MEM, txSlice.offset + baseOffset_, srcMem.ptr(), txSlice.size, stream_));
    return HCCL_SUCCESS;
}

HcclResult AllGatherRing::Rx(const LINK &link, const Slice &rxSlice)
{
    DeviceMem dstMem = outputMem_.range(rxSlice.offset, rxSlice.size);
    HCCL_DEBUG("rx dstMem[%p] range[%llu], size[%llu] ",  dstMem.ptr(),
        rxSlice.offset, rxSlice.size);
    CHK_RET(link->RxAsync(UserMemType::OUTPUT_MEM, rxSlice.offset + baseOffset_, dstMem.ptr(), rxSlice.size, stream_));
    return HCCL_SUCCESS;
}

HcclResult AllGatherRing::RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK> &links)
{
    CHK_SMART_PTR_NULL(dispatcher_);
    CHK_PTR_NULL(stream_.ptr());
    HCCL_INFO("AllGatherRing run_async rank[%u] ranksize[%u] inputMem[%p] outputMem[%p] count[%llu]", \
              rank, rankSize, inputMem_.ptr(), outputMem_.ptr(), count_);

    if (rankSize == 1) {
        if (inputMem_ != outputMem_) {
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, outputMem_, inputMem_, stream_));
        }
        return HCCL_SUCCESS;
    }

    // Algorithm-specific communication link setup
    // Generic communication pattern 
    if (links.size() < rankSize) {
        HCCL_ERROR("[AllGatherRing][RunAsync]rank[%u] linkSize is less than rankSize", rank);
        return HCCL_E_INTERNAL;
    }

    u32 unitSize = DataUnitSize(dataType_);
    if (unitSize == 0) {
        HCCL_ERROR("[AllGatherRing][RunAsync]unitSize is zero");
        return HCCL_E_INTERNAL;
    }

    std::vector<Slice> inputSlices(slices_);
    if (slices_.size() == 0) {
        slices_.resize(rankSize);
        inputSlices.resize(rankSize);

        u64 sliceSize = count_ * unitSize;
        for (u32 i = 0; i < rankSize; i++) {
            slices_[i].size = sliceSize;
            slices_[i].offset = sliceSize * i;
            inputSlices[i].size = sliceSize;
            inputSlices[i].offset = (inputMem_.size() < outputMem_.size()) ? 0 : (sliceSize * i);
            HCCL_DEBUG("rank[%u], slices[%u].offset=%llu, slices[%u].size=%llu", \
                       rank, i, slices_[i].offset, i, slices_[i].size);
        }
    }

    // Copy input to output buffer at correct position
    if (inputMem_ != outputMem_) {
        DeviceMem dst = outputMem_.range(slices_[rank].offset, slices_[rank].size);
        DeviceMem src = inputMem_.range(inputSlices[rank].offset, inputSlices[rank].size);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, stream_));
    }

    CHK_RET(RunAllgather(rank, rankSize, slices_, links));

    if (barrierSwitchOn_) {
        // For non-ring algorithms, barrier is handled in algorithm implementation
    }

    HCCL_INFO("AllGatherRing finished: rank[%u] end", rank);
    return HCCL_SUCCESS;
}

// Core algorithm implementation
HcclResult AllGatherRing::RunAllGather(u32 rank, u32 rankSize, const std::vector<Slice> &outputSlices, const std::vector<LINK> &links)
{
    if (outputSlices.size() < rankSize) {
        HCCL_ERROR("[Run][AllGather]rank[%u] OutputSlice Size is less than rank size", rank);
        return HCCL_E_INTERNAL;
    }
    HcclResult ret = HCCL_SUCCESS;

    // DSL-generated algorithm implementation
        u32 unitSize = DataUnitSize(dataType_);
    if (unitSize == 0) {
        HCCL_ERROR("[AllGatherRing][RunAsync]unitSize is zero");
        return HCCL_E_INTERNAL;
    }
    u64 sliceSize = count_ * unitSize;

    // Ring AllGather Algorithm Implementation
    // Dynamic ring topology with peer-based communication
    
    for (u32 step = 0; step < rankSize - 1; step++) {
        // Ring communication: send to next rank, receive from previous rank
        u32 sendPeer = (rank + 1) % rankSize;
        u32 recvPeer = (rank - 1 + rankSize) % rankSize;
        
        // Chunk forwarding pattern in ring
        u32 sendChunkIdx = (rank - step + rankSize) % rankSize;
        u32 recvChunkIdx = (rank - step - 1 + rankSize) % rankSize;
        
        // Prepare memory for send and receive operations
        u64 chunkSize = sliceSize;
        DeviceMem srcMem = outputMem_.range(sendChunkIdx * chunkSize, chunkSize);
        DeviceMem dstMem = outputMem_.range(recvChunkIdx * chunkSize, chunkSize);
        
        // Dynamic link selection for ring peers
        if (sendPeer >= links.size() || recvPeer >= links.size()) {
                HCCL_ERROR("[AllGatherRing][Ring] peer out of bounds: send[%u] recv[%u] linkSize[%zu]",
                    sendPeer, recvPeer, links.size());
                return HCCL_E_INTERNAL;
        }
        
        // Asynchronous ring communication
        CHK_RET(links[recvPeer]->TxAck(stream_));
        CHK_RET(links[sendPeer]->RxAck(stream_));
        
        CHK_RET(links[sendPeer]->TxAsync(UserMemType::OUTPUT_MEM,
            sendChunkIdx * chunkSize + baseOffset_, srcMem.ptr(), chunkSize, stream_));
        CHK_RET(links[recvPeer]->RxAsync(UserMemType::OUTPUT_MEM,
            recvChunkIdx * chunkSize + baseOffset_, dstMem.ptr(), chunkSize, stream_));
        
        // Wait for communication completion
        CHK_RET(links[recvPeer]->RxWaitDone(stream_));
        CHK_RET(links[sendPeer]->TxWaitDone(stream_));
    }

    return HCCL_SUCCESS;
}

REGISTER_TEMPLATE(TemplateType::TEMPLATE_ALL_GATHER_RING, AllGatherRing);
}  // namespace hccl