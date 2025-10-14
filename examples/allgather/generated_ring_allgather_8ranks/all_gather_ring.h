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

#ifndef ALLGATHER_RING_H
#define ALLGATHER_RING_H

#include "alg_template_base_pub.h"

namespace hccl {
class AllGatherRing : public AlgTemplateBase {
public:
    explicit AllGatherRing(const HcclDispatcher dispatcher);
    ~AllGatherRing() override;

    HcclResult RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK> &links) override;

private:
    // Core algorithm implementation
    HcclResult RunAllGather(u32 rank, u32 rankSize, const std::vector<Slice> &outputSlices, const std::vector<LINK> &links);
    
    // Communication primitives
    HcclResult TxVector(const LINK &link, const std::vector<Slice> &txSlices);
    HcclResult RxVector(const LINK &link, const std::vector<Slice> &rxSlices);
    HcclResult Tx(const LINK &link, const Slice &txSlice);
    HcclResult Rx(const LINK &link, const Slice &rxSlice);
    
    // Utility functions
    inline u32 ForwordRank(u32 rank, u32 rankSize, u32 preNum) const {
        return (rank + rankSize - preNum) % rankSize;
    }
    
    // Communication links
    std::shared_ptr<Transport> linkLeft_;
    std::shared_ptr<Transport> linkRight_;
};
}  // namespace hccl

#endif /* ALLGATHER_RING_H */