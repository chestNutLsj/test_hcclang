# SPDX-License-Identifier: GPL-2.0-only

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .topo_tools import Topology

from fractions import Fraction
import subprocess

def nvidia_gpu():
    # This is a placeholder for a generic Nvidia GPU topology.
    # The actual topology will be defined later.
    # For now, we'll create a simple fully-connected topology for 8 GPUs.
    links = [[1] * 8 for _ in range(8)]
    for i in range(8):
        links[i][i] = 0
    return Topology('NvidiaGPU', links)

# def dgx1():
#     # (0 1 2 3) (4 5 6 7) are two sockets
#     # 0 1 3 2 is the high bandwidth chain in socket 1
#     # 4 5 7 6 is the high bandwidth chain in socket 2
#     # 0 4 and 2 6 are high bandwidth intersocket links  

#     links = [
#         #0  1  2  3  4  5  6  7
#         [0, 2, 1, 1, 2, 0, 0, 0],
#         [2, 0, 1, 2, 0, 1, 0, 0],
#         [1, 1, 0, 2, 0, 0, 2, 0],
#         [1, 2, 2, 0, 0, 0, 0, 1],
#         [2, 0, 0, 0, 0, 2, 1, 1],
#         [0, 1, 0, 0, 2, 0, 1, 2],
#         [0, 0, 2, 0, 1, 1, 0, 2],
#         [0, 0, 0, 1, 1, 2, 2, 0]
#     ]
#     return Topology('DGX1', links)

def h20_cluster(num_nodes=16, num_gpus_per_node=8):
    """
    创建基于NVIDIA H20 GPU的集群拓扑，支持128卡(16节点)配置。

    H20架构特点：
    - 机内拓扑：8个GPU通过NVSwitch 450GB/s全连接，并通过PCIe连接到2个CPU
    - 机间拓扑：每个GPU有2个BF3 NIC，通过ToR交换机连接到聚合层
    - 聚合层：60个聚合交换机，每个120x400G上行，提供48Tbps聚合能力
    - 总体架构：Spine-Leaf式三层网络拓扑

    Args:
        num_nodes (int): 节点数量，默认16个节点
        num_gpus_per_node (int): 每个节点的GPU数量，默认8个GPU

    Returns:
        Topology: H20集群拓扑对象
    """
    num_gpus = num_nodes * num_gpus_per_node
    links = [[0] * num_gpus for _ in range(num_gpus)]
    switches = []

    # H20架构带宽参数 (基于H20拓扑图分析)
    # 机内：NVSwitch 450GB/s
    nvswitch_bw = 450  # GB/s
    
    # 机间：BF3 2x200Gbps = 400Gbps = 50GB/s per GPU
    bf3_bw_per_gpu = 50  # GB/s (400Gbps / 8)
    
    # ToR交换机：8*15=120 ToR，每个ToR 60x400G上行
    tor_uplink_bw = 60 * 50  # 60端口 * 400Gbps = 3000GB/s per ToR
    
    # 聚合层：60个Agg交换机，120x400G上行
    agg_uplink_bw = 120 * 50  # 120端口 * 400Gbps = 6000GB/s per Agg

    # 1. 机内NVSwitch全连接
    for node_idx in range(num_nodes):
        start_gpu = node_idx * num_gpus_per_node
        end_gpu = start_gpu + num_gpus_per_node
        node_gpu_indices = list(range(start_gpu, end_gpu))
        
        # H20节点内所有GPU通过NVSwitch全连接
        for i in range(start_gpu, end_gpu):
            for j in range(start_gpu, end_gpu):
                if i != j:
                    links[i][j] = nvswitch_bw

        # 节点内NVSwitch聚合带宽约束
        # 每个节点的NVSwitch总带宽 = 8 GPU * 450GB/s = 3600GB/s
        nvswitch_aggregate_bw = num_gpus_per_node * nvswitch_bw
        switches.append((
            node_gpu_indices,  # sources
            node_gpu_indices,  # destinations
            nvswitch_aggregate_bw,  # 3600GB/s
            f'Node{node_idx}_NVSwitch_450GB'
        ))

    # 2. 机间网络连接
    # 基于H20的三层网络架构：GPU -> ToR -> Agg -> Spine
    if num_nodes > 1:
        # 每个GPU的机间连接带宽为BF3 NIC带宽
        for node1_idx in range(num_nodes):
            for node2_idx in range(num_nodes):
                if node1_idx == node2_idx:
                    continue
                    
                start1 = node1_idx * num_gpus_per_node
                end1 = start1 + num_gpus_per_node
                start2 = node2_idx * num_gpus_per_node
                end2 = start2 + num_gpus_per_node
                
                # 机间通过BF3 NIC连接
                for i in range(start1, end1):
                    for j in range(start2, end2):
                        links[i][j] = bf3_bw_per_gpu

        # 3. ToR级交换机约束
        # 假设每个节点连接到一个ToR交换机
        for node_idx in range(num_nodes):
            node_start = node_idx * num_gpus_per_node
            node_gpu_indices = list(range(node_start, node_start + num_gpus_per_node))
            all_other_gpus = [gpu for gpu in range(num_gpus) if gpu not in node_gpu_indices]
            
            if all_other_gpus:  # 只有在有其他节点时才添加ToR交换机
                # 每个ToR的上行带宽限制
                switches.append((
                    node_gpu_indices,  # 当前节点的GPU
                    all_other_gpus,    # 其他所有GPU
                    tor_uplink_bw,     # ToR上行带宽 3000GB/s
                    f'ToR{node_idx}_Uplink_60x400G'
                ))

        # 4. 聚合层交换机约束
        # 假设使用segment分组：每8个节点形成一个segment
        gpus_per_segment = 8 * num_gpus_per_node  # 64个GPU per segment
        num_segments = (num_gpus + gpus_per_segment - 1) // gpus_per_segment
        
        for seg_idx in range(num_segments):
            seg_start = seg_idx * gpus_per_segment
            seg_end = min(seg_start + gpus_per_segment, num_gpus)
            segment_gpus = list(range(seg_start, seg_end))
            
            if len(segment_gpus) > num_gpus_per_node:  # 只有多节点segment需要聚合交换机
                switches.append((
                    segment_gpus,  # segment内所有GPU
                    segment_gpus,  # segment内所有GPU
                    agg_uplink_bw, # 聚合交换机带宽 6000GB/s
                    f'AggSwitch_Segment{seg_idx}_120x400G'
                ))

    return Topology(f'NVIDIA-H20-Cluster-{num_nodes}x{num_gpus_per_node}', links, switches)

def dgx_superpod(num_nodes=16, num_gpus_per_node=8, intra_node_bw=900, inter_node_bw=200):
    """
    Models a legacy DGX SuperPOD-like topology for backwards compatibility.
    """
    num_gpus = num_nodes * num_gpus_per_node
    links = [[0] * num_gpus for _ in range(num_gpus)]

    # Intra-node connections (fully connected)
    for node_idx in range(num_nodes):
        start_gpu = node_idx * num_gpus_per_node
        end_gpu = start_gpu + num_gpus_per_node
        for i in range(start_gpu, end_gpu):
            for j in range(i + 1, end_gpu):
                links[i][j] = links[j][i] = intra_node_bw

    # Inter-node connections (fully connected between nodes)
    for node1_idx in range(num_nodes):
        for node2_idx in range(node1_idx + 1, num_nodes):
            start1 = node1_idx * num_gpus_per_node
            end1 = start1 + num_gpus_per_node
            start2 = node2_idx * num_gpus_per_node
            end2 = start2 + num_gpus_per_node
            
            for i in range(start1, end1):
                for j in range(start2, end2):
                    links[i][j] = links[j][i] = inter_node_bw

    return Topology(f'NVIDIA-DGX-SuperPOD-{num_nodes}x{num_gpus_per_node}', links)

def h20_128gpu():
    """
    提供标准的128GPU H20集群拓扑配置。
    等价于16个节点，每个节点8个H20 GPU。
    """
    return h20_cluster(num_nodes=16, num_gpus_per_node=8)

def h20(num_gpus: int = 128):
    """
    获取指定规模的H20拓扑切片。

    Args:
        num_gpus (int): GPU总数，必须是8的倍数，默认128

    Returns:
        Topology: H20拓扑对象
    """
    if num_gpus % 8 != 0:
        raise ValueError(f"GPU总数必须是8的倍数，但收到了 {num_gpus}")
    
    num_nodes = num_gpus // 8
    return h20_cluster(num_nodes=num_nodes, num_gpus_per_node=8)

def dgx_a100():
    """
    Provides a default topology for a 128-GPU DGX SuperPOD.
    This is equivalent to 16 nodes of 8 A100 GPUs each.
    """
    return dgx_superpod(num_nodes=16, num_gpus_per_node=8)

# def nvlink_only(nvidia_smi_topo=None):
#     if nvidia_smi_topo == None:
#         nvidia_smi_topo = _get_nvidia_smi_topo()
#     links = _parse_nvidia_smi_topo(nvidia_smi_topo)
#     return Topology('NVLinkOnly', links)

# def _get_nvidia_smi_topo():
#     output = subprocess.check_output("nvidia-smi topo -m".split())
#     return output.decode("utf-8")

# def _parse_nvidia_smi_topo(output):
#     lines = output.splitlines()
#     before_legend = []
#     for l in lines[1:]:
#         if l and l.startswith("GPU"):
#             # Only look at the rows for GPU
#             before_legend.append(l)
#         else:
#             break
#     devices = [x.split("	")[0] for x in before_legend]
#     gpus = [i for i in range(len(before_legend))
#             if before_legend[i].startswith("GPU")]
#     matrix = [x.split("	")[1:] for x in before_legend]
#     nvlink_matrix = [[_nvlink_num(x[g]) for g in gpus] for x in matrix]
#     return nvlink_matrix

# def _nvlink_num(x):
#     x = x.strip()
#     if x.startswith("NV"):
#         return int(x[2:])
#     else:
#         return 0