/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "alltoallv_alltoallv.h"

namespace hccl {
AlltoAllVNew::AlltoAllVNew(const HcclDispatcher dispatcher)
    : AlgTemplateBase(dispatcher)
{
}

AlltoAllVNew::~AlltoAllVNew() {}

//将输入的 stream 组（子流 subStreams）和配套的通知信号（main ↔ sub stream）进行分类分配，绑定到 SDMA、本地传输的子任务流中，完成内部调度资源准备。
HcclResult AlltoAllVNew::GenerateSubStreamInfo(const std::vector<Stream> &subStreams,
    const std::vector<std::shared_ptr<LocalNotify>> &meshSignalMainToSub,
    const std::vector<std::shared_ptr<LocalNotify>> &meshSignalSubToMain)
{
    u32 totalSubstreamSize = sdmaConcurrentNum_;  //单超节点，只需要SDMA主流
    if (subStreams.size() < totalSubstreamSize || meshSignalMainToSub.size() < totalSubstreamSize ||
        meshSignalSubToMain.size() < totalSubstreamSize) {
        HCCL_ERROR("[AlltoAllVNew][GenerateSubStreamInfo]subStreamsSize[%zu], meshSignalMainToSubSize[%zu]"\
            "meshSignalSubToMainSize[%zu] is smaller than totalSubstreamSize[%u]",subStreams.size(),
            meshSignalMainToSub.size(), meshSignalSubToMain.size(), totalSubstreamSize);
        return HCCL_E_PARA;
    }
    CHK_PRT_RET(links_.size() < userRankSize_, HCCL_ERROR("[AlltoAllVNew][GenerateSubStreamInfo]"\
        "links_.size()[%zu] is smaller than userRankSize_[%u].", links_.size(), userRankSize_),
        HCCL_E_PARA);
    HCCL_DEBUG("subStreams.size[%zu], meshSignalMainToSub.size[%zu], links_.size[%zu]",
        subStreams.size(), meshSignalMainToSub.size(), links_.size());
    u32 index = 0;
    //分配SDMA通道资源
    for (u32 sdmaIndex = 0; sdmaIndex < sdmaConcurrentNum_; sdmaIndex++) {
        sdmaSubStream_.push_back(subStreams[index]);
        sdmaMeshSignalMainToSub_.push_back(meshSignalMainToSub[index]);
        sdmaMeshSignalSubToMain_.push_back(meshSignalSubToMain[index]);
        index++;
    }
    //分配本地控制流
    for (u32 localIndex = 0; localIndex < sdmaConcurrentNum_; localIndex++) {
        localSubStream_.push_back(subStreams[index]);
        localSignalMainToSub_.push_back(meshSignalMainToSub[index]);
        localSignalSubToMain_.push_back(meshSignalSubToMain[index]);
        index++;
    }
    return HCCL_SUCCESS;
}

//通信开始前准备各种资源，并根据当前拓扑、rank位置、数据量等信息推导出通信并发度、数据块划分大小等关键调度参数
HcclResult AlltoAllVNew::Prepare(PrepareData &param)
{
    mainStream_ = param.stream;
    userRank_ = param.userRank;
    userRankSize_ = param.userRankSize;
    links_ = *param.linksPtr;
    localSendRecvInfoPtr_ = param.localSendRecvInfoPtr;
    devNumInlocalPod_ = param.devNumInlocalPod;
    rankIdxInPod_ = param.rankIdxInPod;
    opType_ = param.opType;
    algOpContext_ = param.algOpContext;
    //推算当前Rank所在Pod的起始和结束Rank ID
    podStartRank_ = userRank_ - rankIdxInPod_;
    podEndRank_ = podStartRank_ + devNumInlocalPod_ - 1;
    //本地SDMA通信的并发流数，最多不超过宏定义
    sdmaConcurrentNum_ = (devNumInlocalPod_ > ALLTOALLV_DIRECT_FULLMESH_SDMA_CONCURRENT_SIZE) ?
        (ALLTOALLV_DIRECT_FULLMESH_SDMA_CONCURRENT_SIZE) : (devNumInlocalPod_);

    CHK_PRT_RET(userRankSize_ == 0, HCCL_ERROR("[AlltoAllVNew][Prepare]userRankSize_ is zero."),
        HCCL_E_PARA);

    userInput_ = param.inputMem;
    userOutput_ = param.outputMem;
    cclInMem_ = param.cclInMem;
    cclOutMem_ = param.cclOutMem;
    workMode_ = param.workMode;
    isSuPodAsym_ = param.isSuPodAsym;
    
    //判断是否为"大数据量"通信,依据是否超过宏定义 ALLTOALLV_DIRECT_FULLMESH_BIG_SIZE
    //如果是大数据量通信，则需要使用并发拷贝的方式进行
    u64 maxSendLen = CalcMaxSendLen();
    isBigCount_ = (maxSendLen > ALLTOALLV_DIRECT_FULLMESH_BIG_SIZE) ? true : false;
    CHK_RET(GenerateSubStreamInfo(*param.subStreamsPtr, *param.signalPtr, *param.signalAuxPtr));

    /* 考虑当group0 的rank 跟 group 1的所有rank通信时，每次都要收发，所以取sdmaConcurrentNum_块；
    跟group 0内的rank通信有一块儿浪费 */
    u32 blockGroup = (isBigCount_ || opType_ == HcclCMDType::HCCL_CMD_ALLTOALLV) ? 2 : 1;
    sdmaDataBlockSize_= (cclInMem_.size() / std::max(1u, sdmaConcurrentNum_ * blockGroup));

    // 向下对齐到16k Byte 保证通信数据块对齐
    if (sdmaDataBlockSize_> HCCL_MIN_SLICE_ALIGN_910B) {
        sdmaDataBlockSize_= (sdmaDataBlockSize_/ HCCL_MIN_SLICE_ALIGN_910B) * HCCL_MIN_SLICE_ALIGN_910B;
    }
    CHK_PRT_RET(sdmaDataBlockSize_== 0, HCCL_ERROR("[AlltoAllVNew][Prepare]sdmaDataBlockSize_is zero."),
        HCCL_E_INTERNAL);
    HCCL_DEBUG("[AlltoAllVNew][Prepare] userRank [%u] total cclsize[%llu]," \
        "sdmaDataBlockSize_[%llu], BigCountFlag[%d], stepSize[%u]", userRank_, cclInMem_.size(), sdmaDataBlockSize_, isBigCount_,
        algOpContext_.mc2Handler.stepSize);

    return HCCL_SUCCESS;
}

//构造一个字符串，表示 subStreamReadInfo_ 中每个目标 rank 所对应的 streamIndex（流索引）列表，用于调试或日志记录。
std::string AlltoAllVNew::GetStreamIndexString()
{
    std::string res = "";
    for (auto& info : subStreamReadInfo_) {
        u32 destRank = info.first;
        u32 streamIndex = destRank % sdmaConcurrentNum_;
        res += std::to_string(streamIndex) + ", ";
    }
    return res;
}

//计算 AllToAllV 通信中当前 rank 向所有目标 rank 发送数据中，最大的一次发送数据长度（maxSendLen）
u64 AlltoAllVNew::CalcMaxSendLen()
{
    u64 maxSendLen = 0;
    const SendRecvInfo& localSendRecvInfo = *localSendRecvInfoPtr_;

    for (u32 dstRank = 0; dstRank < localSendRecvInfo.sendLength.size(); dstRank++) {
        maxSendLen = std::max(maxSendLen, localSendRecvInfo.sendLength[dstRank]);
    }

    HCCL_DEBUG("[AlltoAllVNew][CalcMaxSendLen] maxSendLen[%llu]", maxSendLen);
    return maxSendLen;
}

//为当前用户 rank 从某个目标 rank（destRank）接收数据时，生成分段读数据的信息（ReadDataBlock）列表，**用于后续执行 SDMA 或内存 copy 操作。
void AlltoAllVNew::UpdateCurrRankRecvInfo(u32 roundIdx, u32 side, u32 destRank,
    std::vector<ReadDataBlock>& readInfo, u32 maxRecvStep)
{
    const SendRecvInfo& localSendRecvInfo = *localSendRecvInfoPtr_;
    u64 remainRecvLen = localSendRecvInfo.recvLength[destRank];
    u64 scratchOffset = 0;
    u32 bufferIdx = 0;
    u32 pairNum = sdmaConcurrentNum_ / RANK_SET_COMPUTE_CONST;
    if (sdmaConcurrentNum_ == 1) { // 保证和当前rank距离一样时，send/recv用的是同一块buff
        bufferIdx = 0;
    } else if (side == 0) { // 在curRank左边
        u32 gap = (userRank_ - destRank + devNumInlocalPod_) % devNumInlocalPod_;
        bufferIdx = pairNum - (gap - roundIdx * pairNum);
    } else if (side == 1) { // 在curRank右边
        u32 gap = (destRank - userRank_ + devNumInlocalPod_) % devNumInlocalPod_;
        bufferIdx = pairNum - 1 + (gap - roundIdx * pairNum);
    } else { // 最后一个中间位置的rank
        bufferIdx = 0;
    }

    if ((isBigCount_ || opType_ == HcclCMDType::HCCL_CMD_ALLTOALLV ) &&
        (roundIdx % RANK_SET_COMPUTE_CONST != 0)) { // 奇数轮，用下半Buffer
        bufferIdx += sdmaConcurrentNum_;
    }

    scratchOffset = bufferIdx * sdmaDataBlockSize_;

    u32 recvStepIdx = 0;
    u64 dataOffset = 0;
    HCCL_DEBUG("usrRank[%u] total recv localSendRecvInfo.recvLength[%llu] from dstRank[%u] bufferIdx[%u]",
        userRank_, remainRecvLen, destRank, bufferIdx);

    while(recvStepIdx < maxRecvStep && remainRecvLen > 0) {
        u64 currDataRemainLen = localSendRecvInfo.recvLength[destRank] - dataOffset;
        u64 recvLen = std::min(sdmaDataBlockSize_, currDataRemainLen);
        u64 userOutOffset = localSendRecvInfo.recvOffset[destRank] + dataOffset;
        HCCL_DEBUG("[AlltoAllVNew][UpdateCurrRankRecvInfo] usrRank[%u] recv from destRank [%u]"
            "sendStepIdx[%u] recvLen[%lu] userOutOffset[%llu] scratchOffset[%llu]",
            userRank_, destRank, recvStepIdx, recvLen, userOutOffset, scratchOffset);
        readInfo.push_back({recvLen, scratchOffset, userOutOffset});
        dataOffset += recvLen;
        recvStepIdx++;
        remainRecvLen -= recvLen;
    }
}

//为当前用户 rank 向某个目标 rank（destRank）发送数据时，生成分段读数据的信息（ReadDataBlock）列表，**用于后续执行 SDMA 或内存 copy 操作。
void AlltoAllVNew::UpdateCurrRankSendInfo(u32 roundIdx, u32 side, u32 destRank,
    std::vector<SendDataBlock>& sendInfo, u32 maxSendStep)
{
    const SendRecvInfo& localSendRecvInfo = *localSendRecvInfoPtr_;
    u64 remainSendLen = localSendRecvInfo.sendLength[destRank];

    u64 scratchOffset = 0;
    u32 bufferIdx = 0;
    u32 pairNum = sdmaConcurrentNum_ / RANK_SET_COMPUTE_CONST;
    if (sdmaConcurrentNum_ == 1) { // 保证和当前rank距离一样时，send/recv用的是同一块buff
        bufferIdx = 0;
    } else if (side == 0) { // 在curRank左边
        u32 gap = (userRank_ - destRank + devNumInlocalPod_) % devNumInlocalPod_;
        bufferIdx = pairNum - 1 + (gap - roundIdx * pairNum);
    } else if (side == 1) { // 在curRank右边
        u32 gap = (destRank - userRank_ + devNumInlocalPod_) % devNumInlocalPod_;
        bufferIdx = pairNum - (gap - roundIdx * pairNum);
    } else { // 最后一个中间位置的rank
        bufferIdx = 0;
    }

    if ((isBigCount_ || opType_ == HcclCMDType::HCCL_CMD_ALLTOALLV ) &&
        (roundIdx % RANK_SET_COMPUTE_CONST != 0)) { // 奇数轮，用下半Buffer
        bufferIdx += sdmaConcurrentNum_;
    }
    scratchOffset = bufferIdx * sdmaDataBlockSize_;

    u32 sendStepIdx = 0;
    u64 dataOffset = 0;
    HCCL_DEBUG("usrRank[%u] total send localSendRecvInfo.sendLength[%llu] to dstRank[%u] bufferIdx[%u]",
        userRank_, remainSendLen, destRank, bufferIdx);

    while (sendStepIdx < maxSendStep && remainSendLen > 0) {
        u64 currDataRemainLen = localSendRecvInfo.sendLength[destRank] - dataOffset;
        u64 sendLen = std::min(sdmaDataBlockSize_, currDataRemainLen);
        u64 userInOffset = localSendRecvInfo.sendOffset[destRank] + dataOffset;
        HCCL_DEBUG("[AlltoAllVNew][UpdateCurrRankSendInfo] usrRank[%u] send to destRank [%u]"
            " sendStepIdx[%u] sendLen[%lu] userInOffset[%llu] scratchOffset[%llu]",
            userRank_, destRank, sendStepIdx, sendLen, userInOffset, scratchOffset);
        sendInfo.push_back({sendLen, userInOffset, scratchOffset});
        dataOffset += sendLen;
        sendStepIdx++;
        remainSendLen -= sendLen;
    }
}

//为当前通信轮次 roundIdx 中参与通信的 peer rank，生成每个子流（subStream）上所需执行的发送（Send）与接收（Recv）任务调度信息，填入两个输出哈希表中
//subStreamReadInfo ：记录从哪些远程 rank 读取数据（ReadDataBlock）
//subStreamSendInfo ：记录向哪些远程 rank 发送数据（SendDataBlock）
//这两个结构将被用于后续调度阶段，驱动每个 subStream 执行真正的拷贝通信。
void AlltoAllVNew::UpdateSendRecvInfo(u32 roundIdx,
    std::unordered_map<u32, std::vector<ReadDataBlock>> &subStreamReadInfo,
    std::unordered_map<u32, std::vector<SendDataBlock>> &subStreamSendInfo,
    const std::vector<std::vector<std::pair<u32,u32>>> &partialCommRankSet)
{
    for (u32 side = 0; side < partialCommRankSet.size(); side++) {
        for (u32 j = 0; j < partialCommRankSet[side].size(); j++) {
            u32 readRemoteRank = partialCommRankSet[side][j].first;
            if (readRemoteRank == userRank_) {
                continue;
            }
            u32 currDestRecvStep = recvNumSubStep_[readRemoteRank];
            std::vector<ReadDataBlock> readInfo;
            UpdateCurrRankRecvInfo(roundIdx, side, readRemoteRank, readInfo, currDestRecvStep);

            subStreamReadInfo[readRemoteRank] = readInfo;
        }
    }

    for (u32 side = 0; side < partialCommRankSet.size(); side++) {
        for (u32 j = 0; j < partialCommRankSet[side].size(); j++) {
            u32 sendRemoteRank = partialCommRankSet[side][j].second;
            if (sendRemoteRank == userRank_) {
                continue;
            }
            u32 currDestSendStep = sendNumSubStep_[sendRemoteRank];
            std::vector<SendDataBlock> sendInfo;
            UpdateCurrRankSendInfo(roundIdx, side, sendRemoteRank, sendInfo, currDestSendStep);

            subStreamSendInfo[sendRemoteRank] = sendInfo;
        }
    }
}

//AlltoAll 算法的 OP_BASE 模式下，为当前轮次（以及下一个轮次）准备好用于调度的数据传输信息
void AlltoAllVNew::UpdateOpBaseSubStreamInfo(u32 roundIdx)
{
    if (roundIdx == 0 || !isBigCount_) {
        subStreamReadInfo_.clear();
        subStreamSendInfo_.clear();
        UpdateSendRecvInfo(roundIdx, subStreamReadInfo_, subStreamSendInfo_, partialCommRankSet_);
    }
    if (isBigCount_ && (roundIdx < commRounds_ - 1)) {
        nextSubStreamReadInfo_.clear();
        nextSubStreamSendInfo_.clear();
        UpdateSendRecvInfo(roundIdx + 1, nextSubStreamReadInfo_, nextSubStreamSendInfo_, nextPartialCommRankSet_);
    }
}

//在 AllToAllV 算法中，将用户输入数据从 userInput_ 拷贝到中间缓存 cclInMem_，用于同一个超节点（intra-node）内的通信。
HcclResult AlltoAllVNew::PrepareIntraData(u32 step,
    std::unordered_map<u32,std::vector<SendDataBlock>> &subStreamSendInfo)
{
    u32 sendDataIndex = 0;
    for (auto& sdmaInfo : subStreamSendInfo) {
        const std::vector<SendDataBlock>& sendInfo = sdmaInfo.second;
        if (step < sendNumSubStep_[sdmaInfo.first]) {
            DeviceMem src = userInput_.range(sendInfo[step].userInOffset, sendInfo[step].sendLen);
            DeviceMem dst = cclInMem_.range(sendInfo[step].scratchOffset, sendInfo[step].sendLen);
            HCCL_DEBUG("[AlltoAllVNew][PrepareIntraData]userRank [%u] copy from userInOffset[%llu]"
                "len[%u] to scratchOffset [%llu]", userRank_, sendInfo[step].userInOffset, sendInfo[step].sendLen,
                sendInfo[step].scratchOffset);
            if (isBigCount_) {
                CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, localSubStream_[sendDataIndex]));
            } else {
                CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, mainStream_));
            }
        }
        sendDataIndex++;
    }
    return HCCL_SUCCESS;
}

//更新每一轮中需要通信的远程rank集合
void AlltoAllVNew::UpdateRemoteRankSet(u32 roundIdx, u32 groupRankSize)
{
    if (sdmaConcurrentNum_ == 1) { 
        //串行通信时，所有数据都在一个流上处理
        UpdatePartialCommunicationRankSetPairWise(roundIdx, groupRankSize);
    } else {
        //并行通信时，分组处理
        UpdatePartialCommunicationRankSet(roundIdx, groupRankSize, partialCommRankSet_);
    }
}

// 单 SDMA 通道下每轮通信对端 rank 生成
//在当前轮 roundIdx 中，为当前 rank 生成一对要通信的 rank（接收和发送对象），并填入 partialCommRankSet_
void AlltoAllVNew::UpdatePartialCommunicationRankSetPairWise(u32 roundIdx, u32 groupRankSize)
{
    partialCommRankSet_.clear();
    partialCommRankSet_.resize(1);
    for (u32 i = roundIdx * sdmaConcurrentNum_; i < (roundIdx * sdmaConcurrentNum_ + groupRankSize); i++) {
        u32 readRemoteRank = podStartRank_ + (rankIdxInPod_ + devNumInlocalPod_ - i) % devNumInlocalPod_;
        u32 sendRemoteRank = podStartRank_ + (rankIdxInPod_ + i) % devNumInlocalPod_;
        partialCommRankSet_[0].push_back(std::make_pair(readRemoteRank, sendRemoteRank));
        HCCL_DEBUG("[AlltoAllVNew][UpdatePartialCommunicationRankSetPairWise] userRank [%u] i[%u]" \
            "readRemoteRank[%u] writeRemoteRank[%u]", userRank_, i, readRemoteRank, sendRemoteRank);
    }
    HCCL_DEBUG("[AlltoAllVNew][UpdatePartialCommunicationRankSetPairWise] partialCommRankSet_ size[%zu]",
        partialCommRankSet_[0].size());
}

//多通道（并发）AllToAllV 通信调度中，每轮通信对端 rank 计算与分组的核心逻辑
//根据当前通信轮次、设备数量、并发通道数，计算并划分了多个通信对集合（rank pair）(recvRank, sendRank)，并根据是否对称放入三类通信集合中（左、右、自发自收
void AlltoAllVNew::UpdatePartialCommunicationRankSet(u32 roundIdx, u32 groupRankSize,
    std::vector<std::vector<std::pair<u32,u32>>> &partialCommRankSet)
{
    partialCommRankSet.clear();
    partialCommRankSet.resize(RANK_SET_COMPUTE_CONST + 1);
    u32 pairNumPerRound = sdmaConcurrentNum_ / RANK_SET_COMPUTE_CONST;
    u32 pairSize = (groupRankSize < sdmaConcurrentNum_) ?
        (groupRankSize + RANK_SET_COMPUTE_CONST - 1) / RANK_SET_COMPUTE_CONST: pairNumPerRound;
    for (u32 i = roundIdx * pairNumPerRound + 1;
         i < (roundIdx * pairNumPerRound + pairSize + 1); i++) {
        u32 leftRemoteRank = podStartRank_ + (rankIdxInPod_ + devNumInlocalPod_ - i) % devNumInlocalPod_;
        u32 rightRemoteRank = podStartRank_ + (rankIdxInPod_ + i) % devNumInlocalPod_;
        if (leftRemoteRank == rightRemoteRank) {
            partialCommRankSet[2].push_back(std::make_pair(leftRemoteRank, leftRemoteRank));
        } else {
            partialCommRankSet[0].push_back(std::make_pair(leftRemoteRank, leftRemoteRank));
            partialCommRankSet[1].push_back(std::make_pair(rightRemoteRank, rightRemoteRank));
        }
        HCCL_DEBUG("[AlltoAllVNew][UpdatePartialCommunicationRankSet] round[%u] userRank [%u] i[%u]" \
            "read/write leftRemoteRank[%u] rightRemoteRank[%u]", roundIdx, userRank_, i, leftRemoteRank, rightRemoteRank);
    }
    HCCL_DEBUG("[AlltoAllVNew][UpdatePartialCommunicationRankSet] round[%u] partialCommRankSet_ total size[%zu]",
        roundIdx, partialCommRankSet[0].size() + partialCommRankSet[1].size() + partialCommRankSet[2].size());
}

// 主流只需要通知当前子步骤需要收发数据的 SDMA 流，减少同步开销
//主流向所有 SDMA 子流发出通知，告知它们可以启动对应子步骤的通信任务，并通过轻量级的同步机制确保调度有序，同时发起一个空任务以避免同步隐式等待。
HcclResult AlltoAllVNew::NotifySubStreamStart()
{
    for (u32 streamIndex = 0; streamIndex < subStreamReadInfo_.size(); streamIndex++) {
        CHK_RET(LocalNotify::Post(mainStream_, dispatcher_, sdmaMeshSignalSubToMain_[streamIndex], INVALID_VALUE_STAGE));
        CHK_RET(LocalNotify::Wait(sdmaSubStream_[streamIndex], dispatcher_, sdmaMeshSignalSubToMain_[streamIndex],
            INVALID_VALUE_STAGE));
    }
    for (u32 streamIndex = 0; streamIndex < subStreamReadInfo_.size(); streamIndex++) {
        CHK_RET(ExecEmptyTask(userInput_, userOutput_, sdmaSubStream_[streamIndex], dispatcher_));
    }
    HCCL_DEBUG("[AlltoAllVNew][NotifySubStreamStart] userRank [%u] main stream notify sdma stream [%s]",
        userRank_, GetStreamIndexString().c_str());
    return HCCL_SUCCESS;
}

//通信完成后的主流等待子流结束，确保所有 SDMA 子流的收发任务都已经完成再继续往下调度。
HcclResult AlltoAllVNew::WaitSubStreamFinish()
{
    for (u32 streamIndex = 0; streamIndex < subStreamReadInfo_.size(); streamIndex++) {
        CHK_RET(LocalNotify::Post(sdmaSubStream_[streamIndex], dispatcher_, sdmaMeshSignalMainToSub_[streamIndex],
            INVALID_VALUE_STAGE));
        CHK_RET(LocalNotify::Wait(mainStream_, dispatcher_, sdmaMeshSignalMainToSub_[streamIndex],
            INVALID_VALUE_STAGE));
    }
    HCCL_DEBUG("[AlltoAllVNew][WaitSubStreamFinish] userRank [%u] main stream wait sdma stream [%s]",
        userRank_, GetStreamIndexString().c_str());
    return HCCL_SUCCESS;
}

//主流通过信号 localSignalSubToMain_ 通知每个 local 子流（如用于机内通信的 stream）可以启动执行通信任务，并确保顺序同步。
HcclResult AlltoAllVNew::NotifyLocalSubStreamStart()
{
    for (u32 streamIndex = 0; streamIndex < subStreamSendInfo_.size(); streamIndex++) {
        CHK_RET(LocalNotify::Post(mainStream_, dispatcher_, localSignalSubToMain_[streamIndex], INVALID_VALUE_STAGE));
        CHK_RET(LocalNotify::Wait(localSubStream_[streamIndex], dispatcher_, localSignalSubToMain_[streamIndex],
            INVALID_VALUE_STAGE));
    }
    return HCCL_SUCCESS;
}

//主流等待所有本地通信子流（localSubStream_[]）完成各自的通信任务（如节点内 memcpy），确保 local AllToAllV 步骤已全部完成，才进入下一步调度
HcclResult AlltoAllVNew::WaitLocalSubStreamFinish()
{
    for (u32 streamIndex = 0; streamIndex < subStreamSendInfo_.size(); streamIndex++) {
        CHK_RET(LocalNotify::Post(localSubStream_[streamIndex], dispatcher_, localSignalMainToSub_[streamIndex],
            INVALID_VALUE_STAGE));
        CHK_RET(LocalNotify::Wait(mainStream_, dispatcher_, localSignalMainToSub_[streamIndex],
            INVALID_VALUE_STAGE));
    }
    return HCCL_SUCCESS;
}

//根据每个目标 rank 的发送与接收数据长度，计算出本 rank 需要执行多少个最大发送/接收子步骤（sub-step），以便在后续调度阶段进行分段通信。
u32 AlltoAllVNew::CalcNumSubStep()
{
    const SendRecvInfo& localSendRecvInfo = *localSendRecvInfoPtr_;

    sendNumSubStep_.clear();
    recvNumSubStep_.clear();
    u32 numSubStep = 0;

    for (u32 destRank = podStartRank_; destRank < podStartRank_ + devNumInlocalPod_; destRank++) {
        if (destRank == userRank_) {
            continue;
        }

        u32 currRankSendSubStep = ((localSendRecvInfo.sendLength[destRank] + sdmaDataBlockSize_- 1) / sdmaDataBlockSize_);
        sendNumSubStep_[destRank] = currRankSendSubStep;

        u32 currRankRecvSubStep = ((localSendRecvInfo.recvLength[destRank] + sdmaDataBlockSize_- 1) / sdmaDataBlockSize_);
        recvNumSubStep_[destRank] = currRankRecvSubStep;
        HCCL_DEBUG("[AlltoAllVNew][CalcNumSubStep] userRank [%u] currRankSendSubStep[%u]" \
        "currRankRecvSubStep[%u]", userRank_, currRankSendSubStep, currRankRecvSubStep);
        numSubStep = std::max(numSubStep, std::max(currRankSendSubStep, currRankRecvSubStep));
    }
    HCCL_DEBUG("[AlltoAllVNew][CalcNumSubStep] userRank [%u] max communication step[%u]",
        userRank_, numSubStep);
    return numSubStep;
}

//对当前通信步骤 step，遍历所有涉及远程通信的 rank 对（recvRank ←→ sendRank），通过 TxAck 和 RxAck 在对应 stream 上向远程 rank 发出 "我要开始收/发数据" 的通知，完成远端通信前的 handshake。
HcclResult AlltoAllVNew::NotifyRemoteRankStart(u32 step)
{
    u32 streamIndex = 0;
    for (auto& sendRecvSide : partialCommRankSet_) {
        for (auto& sendRecvPair : sendRecvSide) {
            u32 recvRank = sendRecvPair.first;
            u32 sendRank = sendRecvPair.second;
            if (sendRank == userRank_) {
                continue;
            }
            const std::vector<ReadDataBlock>& readInfo = subStreamReadInfo_[recvRank];
            const std::vector<SendDataBlock>& sendInfo = subStreamSendInfo_[sendRank];
            Stream& currStream = sdmaSubStream_[streamIndex];
            const LINK& readTransport = links_[recvRank];
            const LINK& sendTransport = links_[sendRank];
            if (step < sendInfo.size()) {
                CHK_RET(sendTransport->TxAck(currStream));
            }

            if (step < readInfo.size()) {
                CHK_RET(readTransport->RxAck(currStream));
            }
            streamIndex ++;
        }
    }
    HCCL_INFO("[AlltoAllVNew][NotifyRemoteRankStart] done");
    return HCCL_SUCCESS;
}

//在第 step 步，遍历所有远程通信对，执行远程内存数据异步拷贝到本地（跨节点 D2D memcpy），并发出数据传输完成的信号通知，完成单步远程通信。
HcclResult AlltoAllVNew::SDMAwithRemoteRankAndNotifyEnd(u32 step)
{
    u32 streamIndex = 0;
    for (auto& sendRecvSide : partialCommRankSet_) {
        for (auto& sendRecvPair : sendRecvSide) {
            u32 recvRank = sendRecvPair.first;
            u32 sendRank = sendRecvPair.second;
            if (sendRank == userRank_) {
                continue;
            }
            const std::vector<ReadDataBlock>& readInfo = subStreamReadInfo_[recvRank];
            const std::vector<SendDataBlock>& sendInfo = subStreamSendInfo_[sendRank];
            Stream& currStream = sdmaSubStream_[streamIndex];
            const LINK& readTransport = links_[recvRank];
            const LINK& sendTransport = links_[sendRank];
            if (step < readInfo.size()) {
                const LINK& intraNeighboorTransport = links_[recvRank];
                CHK_PTR_NULL(intraNeighboorTransport);
                void* remDMAMemPtr = nullptr;
                CHK_RET(intraNeighboorTransport->GetRemoteMem(UserMemType::INPUT_MEM, &remDMAMemPtr));
                DeviceMem remoteCCLInMem = DeviceMem::create(static_cast<u8 *>(remDMAMemPtr), cclInMem_.size());
                DeviceMem srcMem = remoteCCLInMem.range(readInfo[step].remoteOffset, readInfo[step].recvLen);
                DeviceMem dstMem = userOutput_.range(readInfo[step].recvOffset, readInfo[step].recvLen);
                CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, currStream,
                    readTransport->GetRemoteRank(), readTransport->GetLinkType()));
                HCCL_DEBUG("[AlltoAllVNew][SendRecvData] userRank [%u], recvRank[%u], sendRank[%u]," \
                    "sdma stream [%llu] read data from remote offset [%llu] len [%llu] to local [%llu]",
                    userRank_,  recvRank, sendRank, streamIndex, readInfo[step].remoteOffset,
                    readInfo[step].recvLen, readInfo[step].recvOffset); 
                CHK_RET(readTransport->TxDataSignal(currStream));
            }
            if (step < sendInfo.size()) {
                CHK_RET(sendTransport->RxDataSignal(currStream));
            }
            streamIndex ++;
        }
    }
    HCCL_INFO("[AlltoAllVNew][SDMAwithRemoteRankAndNotifyEnd] done");
    return HCCL_SUCCESS;
}

//按照当前 step 和轮次 roundIdx，完成跨节点通信信号通知、子流同步、数据拷贝准备和远程 SDMA 异步数据传输，实现分步分轮次的 AllToAllV 通信。
HcclResult AlltoAllVNew::SendRecvData(u32 step, u32 roundIdx)
{
    HCCL_DEBUG("[AlltoAllVNew][SendRecvData] userRank [%u] sdma stream [%s] wait main stream",
        userRank_, GetStreamIndexString().c_str());
    CHK_RET(NotifyRemoteRankStart(step));
    CHK_RET(WaitSubStreamFinish());
    CHK_RET(ExecEmptyTask(userInput_, userOutput_, mainStream_, dispatcher_));
    CHK_RET(NotifySubStreamStart());
    if (isBigCount_ && (roundIdx < commRounds_ - 1)) {
        CHK_RET(NotifyLocalSubStreamStart());
        CHK_RET(PrepareIntraData(step, nextSubStreamSendInfo_));
    }
    CHK_RET(SDMAwithRemoteRankAndNotifyEnd(step));

    return HCCL_SUCCESS;
}

//将当前 rank 在用户输入缓冲区中指定的"发给自己"的数据区域，异步复制到对应的输出缓冲区中。
HcclResult AlltoAllVNew::LocalCopy()
{
    const SendRecvInfo& localSendRecvInfo = *localSendRecvInfoPtr_;
    DeviceMem src = userInput_.range(localSendRecvInfo.sendOffset[userRank_],
        localSendRecvInfo.sendLength[userRank_]);
    DeviceMem dst = userOutput_.range(localSendRecvInfo.recvOffset[userRank_],
        localSendRecvInfo.recvLength[userRank_]);
    HCCL_DEBUG("[AlltoAllVNew][LocalCopy]userRank [%u] copy from userInput [%llu] len [%llu]" \
        "to userOutput [%llu] dstLen[%llu]", userRank_, localSendRecvInfo.sendOffset[userRank_],
        localSendRecvInfo.sendLength[userRank_],
        localSendRecvInfo.recvOffset[userRank_],
        localSendRecvInfo.recvLength[userRank_]);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, mainStream_));

    return HCCL_SUCCESS;
}

//该函数协调本地和远程通信的准备、同步、数据发送和本地拷贝，实现多轮次多步的 AlltoAllV 通信调度。
HcclResult AlltoAllVNew::RunGroupFullMeshAlltoall(u32 roundIdx, u32 step)
{
    UpdateOpBaseSubStreamInfo(roundIdx);
    CHK_RET(ExecEmptyTask(userInput_, userOutput_, mainStream_, dispatcher_));
    if (isBigCount_ && (roundIdx == 0) ) {
        CHK_RET(NotifyLocalSubStreamStart());
        CHK_RET(PrepareIntraData(step, subStreamSendInfo_));
        CHK_RET(WaitLocalSubStreamFinish());
        CHK_RET(ExecEmptyTask(userInput_, userOutput_, mainStream_, dispatcher_));
    } else if (!isBigCount_) {
        CHK_RET(PrepareIntraData(step, subStreamSendInfo_));
    }
    CHK_RET(NotifySubStreamStart());
    CHK_RET(ExecEmptyTask(userInput_, userOutput_, mainStream_, dispatcher_));
    CHK_RET(SendRecvData(step, roundIdx));
    if (step == 0 && !islocalCpyDone_) {
        CHK_RET(LocalCopy());
        islocalCpyDone_ = true;
    }
    CHK_RET(ExecEmptyTask(userInput_, userOutput_, mainStream_, dispatcher_));
    CHK_RET(WaitSubStreamFinish());
    if (isBigCount_ && (roundIdx < commRounds_ - 1)) {
        CHK_RET(WaitLocalSubStreamFinish());
    }
    CHK_RET(ExecEmptyTask(userInput_, userOutput_, mainStream_, dispatcher_));
    return HCCL_SUCCESS;
}

//根据当前轮次 roundIdx 和组内规模 groupRankSize，更新参与通信的远端 rank 集合（通信对）；
//调用实际执行函数 RunGroupFullMeshAlltoall 完成该轮次该步骤的通信调度和执行；
//在大数据分轮次模式下，准备和切换下一轮的通信信息和状态；
//最后启动本地子流任务，完成本地异步通信调度。
HcclResult AlltoAllVNew::RunSDMATasks(u32 roundIdx, u32 step, u32 groupRankSize, u32 leftRankSize)
{
    if (isBigCount_) {
        if (roundIdx == 0) {
            UpdatePartialCommunicationRankSet(roundIdx, groupRankSize, partialCommRankSet_);
        }
        if (roundIdx < commRounds_ - 1) {
            u32 nextgroupRankSize = (leftRankSize - groupRankSize > sdmaConcurrentNum_) ?
                sdmaConcurrentNum_ : leftRankSize - groupRankSize;
            UpdatePartialCommunicationRankSet(roundIdx + 1, nextgroupRankSize, nextPartialCommRankSet_);
        }
        CHK_RET(RunGroupFullMeshAlltoall(roundIdx, step));

        if (roundIdx < commRounds_ - 1) {
            partialCommRankSet_ = nextPartialCommRankSet_;
            subStreamSendInfo_ = nextSubStreamSendInfo_;
            subStreamReadInfo_ = nextSubStreamReadInfo_;
        }
        CHK_RET(LaunchTaskExtend(dispatcher_, mainStream_, localSubStream_));
    } else {
        UpdatePartialCommunicationRankSet(roundIdx, groupRankSize, partialCommRankSet_);
        CHK_RET(RunGroupFullMeshAlltoall(roundIdx, step));
    }
    return HCCL_SUCCESS;
}

//执行 SDMA 通信调度，计算通信轮次和子步骤，处理细粒度通信和大数据分轮次通信的逻辑，确保正确的通信顺序和数据传输。
//如果是细粒度通信，调用 RunSDMAFineGrained；否则，按照总步骤和轮次进行分组通信，调用 RunSDMATasks 完成每个子步骤的通信任务调度。
//在每个通信步骤中，执行通知、等待、数据准备和远程数据传输，确保所有通信任务按顺序完成。
//最后，记录通信完成的状态，并返回成功结果。
HcclResult AlltoAllVNew::RunSDMA(HcclOpMetaInfoDef &opMeta)
{
    u32 totalStep = CalcNumSubStep();
 
    // 计算每个rank分组fullmesh后需要通信的轮次，向上取整
    commRounds_ = (devNumInlocalPod_ + sdmaConcurrentNum_ - 1) / sdmaConcurrentNum_;
    HCCL_ERROR("[AlltoAllVNew][RunSDMA] userRank [%u] communication rounds[%llu] totalStep %u stepSize %u",
        userRank_, commRounds_, totalStep, algOpContext_.mc2Handler.stepSize);

    if (totalStep == 0 && !islocalCpyDone_) {
        CHK_RET(InitTask(dispatcher_, mainStream_, opMeta.isEnableCache, opMeta.GetCacheKey()));
        CHK_RET(LocalCopy());
        islocalCpyDone_ = true;
        CHK_RET(LaunchTaskExtend(dispatcher_, mainStream_, sdmaSubStream_));
        return HCCL_SUCCESS;
    }

    for (u32 step = 0; step < totalStep; step++) {
        u32 leftRankSize = devNumInlocalPod_ - 1; // leftRankSize中去掉本卡
        for (u32 roundIdx = 0; roundIdx < commRounds_ && leftRankSize > 0; roundIdx++) {
            CHK_RET(InitTask(dispatcher_, mainStream_, opMeta.isEnableCache, opMeta.GetCacheKey()));
            u32 groupRankSize = (leftRankSize > sdmaConcurrentNum_) ? sdmaConcurrentNum_ : leftRankSize;
            CHK_RET(RunSDMATasks(roundIdx, step, groupRankSize, leftRankSize));
            leftRankSize -= groupRankSize;
            CHK_RET(LaunchTaskExtend(dispatcher_, mainStream_, sdmaSubStream_));
        }
    }
    
    HCCL_INFO("[AlltoAllVNew][RunSDMA] finished.");
    return HCCL_SUCCESS;
}

HcclResult AlltoAllVNew::RunAsync()
{   
    HcclOpMetaInfoDef opMeta = HcclOpMetaInfo::GetOneForAllToAllV(CopyPattern::ZCOPY, cclInMem_.size(), true);
    CHK_RET(InitTask(dispatcher_, mainStream_, opMeta.isEnableCache, opMeta.GetCacheKey()));

    if (userRankSize_ == 1) {
        HCCL_INFO("[AlltoAllVNew][RunAsync] do localcopy with 1 rank");
        CHK_RET(LocalCopy());
        return HCCL_SUCCESS;
    }

    CHK_RET(ExecEmptyTask(userInput_, userOutput_, mainStream_, dispatcher_));

    if (devNumInlocalPod_ > 1) {
        CHK_RET(RunSDMA(opMeta));
    }
    HCCL_INFO("[AlltoAllVNew][RunAsync] finished.");
    return HCCL_SUCCESS;
}

REGISTER_TEMPLATE(TemplateType::TEMPLATE_ALL_2_ALL_V_NEW, AlltoAllVNew);
} // namespace hccl
