# SPDX-License-Identifier: GPL-2.0-only

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .topo_tools import Topology
import math
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import itertools

@dataclass
class CM384PhysicalConstants:
    """CM384物理系统常量 - 这些是整个CM384超节点的固有属性"""
    # 物理系统规格
    TOTAL_NODES = 48
    NPUS_PER_NODE_NORMAL = 8      # 正常模式：8个910C NPU
    NPUS_PER_NODE_SPLIT_DIE = 16  # 分die模式：16个逻辑NPU
    TOTAL_NPUS_NORMAL = TOTAL_NODES * NPUS_PER_NODE_NORMAL      # 384
    TOTAL_NPUS_SPLIT_DIE = TOTAL_NODES * NPUS_PER_NODE_SPLIT_DIE # 768
    
    # UB Switch架构
    UB_SWITCHES_PER_NODE = 7
    UB_BW_PER_SWITCH = 56  # GB/s
    
    # Sub-plane架构
    NUM_SUBPLANES = 7
    SWITCHES_PER_SUBPLANE = 16
    SUBPLANE_BW_PER_SWITCH = 28  # GB/s
    
    # 固定的系统级带宽 - 这是整个CM384超节点的物理属性
    TOTAL_INTER_FABRIC_BW = TOTAL_NODES * SUBPLANE_BW_PER_SWITCH  # 1344 GB/s
    
    # 分die模式性能衰减
    SPLIT_DIE_BW_FACTOR = 0.88    # 带宽衰减到88%
    SPLIT_DIE_LATENCY_FACTOR = 1.12  # 延迟增加12%

class CM384SubPlaneRouter:
    """CM384 Sub-plane路由器 - 实现基于Sub-plane的精确路由"""
    
    def __init__(self, constants: CM384PhysicalConstants):
        self.constants = constants
        # 初始化Sub-plane连接矩阵
        self.subplane_connectivity = self._initialize_subplane_connectivity()
        
    def _initialize_subplane_connectivity(self) -> Dict[int, Dict[int, List[int]]]:
        """
        初始化Sub-plane连接矩阵
        每个Sub-plane连接特定的节点集合，模拟真实的CM384路由拓扑
        """
        connectivity = {}
        
        for sp_id in range(self.constants.NUM_SUBPLANES):
            connectivity[sp_id] = {}
            
            # 每个Sub-plane连接所有节点，但连接密度和路径不同
            # 使用确定性哈希来分配节点到Sub-plane的连接强度
            for node_id in range(self.constants.TOTAL_NODES):
                # 计算该节点在此Sub-plane中的连接权重
                hash_val = (node_id * 7 + sp_id * 11) % 100
                if hash_val < 70:  # 70%的节点与此Sub-plane有强连接
                    connectivity[sp_id][node_id] = [0, 1, 2, 3]  # 多个交换机路径
                elif hash_val < 90:  # 20%的节点有中等连接
                    connectivity[sp_id][node_id] = [0, 1]       # 部分交换机路径
                else:  # 10%的节点有弱连接
                    connectivity[sp_id][node_id] = [0]          # 单个交换机路径
                    
        return connectivity
    
    def get_shared_subplanes(self, node1: int, node2: int) -> List[int]:
        """获取两个节点共享的Sub-plane列表"""
        if node1 >= self.constants.TOTAL_NODES or node2 >= self.constants.TOTAL_NODES:
            return []
        
        shared = []
        for sp_id in range(self.constants.NUM_SUBPLANES):
            if (node1 in self.subplane_connectivity[sp_id] and 
                node2 in self.subplane_connectivity[sp_id]):
                shared.append(sp_id)
        return shared
    
    def calculate_inter_node_bandwidth(self, node1: int, node2: int) -> float:
        """
        计算两个节点间的实际带宽
        基于共享Sub-plane、路径多样性和拥塞情况
        """
        if node1 == node2:
            return 0.0
        
        shared_subplanes = self.get_shared_subplanes(node1, node2)
        if not shared_subplanes:
            return 0.0
        
        total_bw = 0.0
        
        for sp_id in shared_subplanes:
            # 计算该Sub-plane在这两个节点间的可用带宽
            node1_paths = len(self.subplane_connectivity[sp_id][node1])
            node2_paths = len(self.subplane_connectivity[sp_id][node2])
            
            # 路径容量 = min(两个节点的路径数) * 每个交换机的带宽
            path_capacity = min(node1_paths, node2_paths) * self.constants.SUBPLANE_BW_PER_SWITCH
            
            # 计算该Sub-plane的总负载（有多少节点对在使用它）
            connected_nodes = len(self.subplane_connectivity[sp_id])
            total_node_pairs = connected_nodes * (connected_nodes - 1)
            
            # 该Sub-plane的总带宽
            subplane_total_bw = (self.constants.SWITCHES_PER_SUBPLANE * 
                               self.constants.SUBPLANE_BW_PER_SWITCH)
            
            # 考虑拥塞：带宽在所有使用该Sub-plane的节点对之间分配
            if total_node_pairs > 0:
                # 但是这里我们还要考虑路径质量
                path_quality = (node1_paths + node2_paths) / 8  # 标准化到[0,1]
                effective_bw = (subplane_total_bw / total_node_pairs) * path_quality
                total_bw += effective_bw
        
        return total_bw

class CM384NodeArchitecture:
    """CM384节点架构 - 实现精确的节点内拓扑"""
    
    def __init__(self, constants: CM384PhysicalConstants):
        self.constants = constants
        
    def get_intra_node_bandwidth(self, npu1: int, npu2: int, node_id: int, 
                                split_die: bool = False) -> float:
        """
        计算节点内两个NPU间的带宽
        考虑UB Switch的路由和910C分die情况
        """
        if npu1 == npu2:
            return 0.0
            
        npus_per_node = (self.constants.NPUS_PER_NODE_SPLIT_DIE if split_die 
                        else self.constants.NPUS_PER_NODE_NORMAL)
        
        # 计算节点内的本地NPU ID
        local_npu1 = npu1 % npus_per_node
        local_npu2 = npu2 % npus_per_node
        
        # 基础UB带宽
        base_bw = self.constants.UB_SWITCHES_PER_NODE * self.constants.UB_BW_PER_SWITCH
        
        if split_die:
            # 分die模式下，检查是否跨die通信
            die1 = local_npu1 // 8  # 每个die 8个NPU
            die2 = local_npu2 // 8
            
            if die1 == die2:
                # 同die内通信，带宽较高
                effective_bw = base_bw * self.constants.SPLIT_DIE_BW_FACTOR
            else:
                # 跨die通信，带宽降低更多
                effective_bw = base_bw * self.constants.SPLIT_DIE_BW_FACTOR * 0.85
        else:
            # 正常模式，所有NPU在同一die中
            effective_bw = base_bw
        
        return effective_bw

class CM384TopologyBuilder:
    """CM384拓扑构建器 - 构建真实的拓扑切片"""
    
    def __init__(self, constants: CM384PhysicalConstants):
        self.constants = constants
        self.router = CM384SubPlaneRouter(constants)
        self.node_arch = CM384NodeArchitecture(constants)
        
    def build_slice_topology(self, slice_nodes: List[int], split_die: bool = False) -> Topology:
        """
        构建CM384切片拓扑
        
        Args:
            slice_nodes: 切片包含的节点列表 (0-47)
            split_die: 是否启用分die模式
        
        Returns:
            Topology: 真实的CM384切片拓扑
        """
        # 验证节点范围
        for node_id in slice_nodes:
            if node_id >= self.constants.TOTAL_NODES:
                raise ValueError(f"节点ID {node_id} 超出范围 [0, {self.constants.TOTAL_NODES-1}]")
        
        npus_per_node = (self.constants.NPUS_PER_NODE_SPLIT_DIE if split_die 
                        else self.constants.NPUS_PER_NODE_NORMAL)
        
        # 构建NPU ID映射
        npu_list = []
        node_to_npus = {}
        
        for i, node_id in enumerate(slice_nodes):
            start_npu = i * npus_per_node  # 在切片中的连续NPU ID
            node_npus = list(range(start_npu, start_npu + npus_per_node))
            npu_list.extend(node_npus)
            node_to_npus[node_id] = node_npus
        
        num_npus = len(npu_list)
        
        # 构建连接矩阵
        links = [[0.0] * num_npus for _ in range(num_npus)]
        
        # 填充连接矩阵
        for i in range(num_npus):
            for j in range(num_npus):
                if i == j:
                    continue
                
                # 找到NPU所属的物理节点
                node_i = slice_nodes[i // npus_per_node]
                node_j = slice_nodes[j // npus_per_node]
                
                if node_i == node_j:
                    # 节点内连接
                    bw = self.node_arch.get_intra_node_bandwidth(i, j, node_i, split_die)
                    links[j][i] = bw
                else:
                    # 节点间连接 - 使用真实的路由计算
                    bw = self.router.calculate_inter_node_bandwidth(node_i, node_j)
                    if split_die:
                        bw *= self.constants.SPLIT_DIE_BW_FACTOR
                    links[j][i] = bw
        
        # 构建交换机约束
        switches = self._build_switches(slice_nodes, node_to_npus, split_die)
        
        # 构建拓扑名称
        die_mode = "SplitDie" if split_die else "Normal"
        topology_name = f'CM384_Slice_{num_npus}NPU_{len(slice_nodes)}Nodes_{die_mode}'
        
        return Topology(topology_name, links, switches)
    
    def _build_switches(self, slice_nodes: List[int], node_to_npus: Dict[int, List[int]], 
                       split_die: bool) -> List[Tuple]:
        """构建交换机约束"""
        switches = []
        
        # 1. 节点内UB交换机
        for node_id in slice_nodes:
            node_npus = node_to_npus[node_id]
            
            # 计算节点内总带宽
            if split_die:
                # 分die模式下，带宽更复杂
                total_bw = (self.constants.UB_SWITCHES_PER_NODE * 
                           self.constants.UB_BW_PER_SWITCH * 
                           len(node_npus) * 
                           self.constants.SPLIT_DIE_BW_FACTOR)
            else:
                total_bw = (self.constants.UB_SWITCHES_PER_NODE * 
                           self.constants.UB_BW_PER_SWITCH * 
                           len(node_npus))
            
            switches.append((
                node_npus,
                node_npus,
                total_bw,
                f'Node{node_id}_UB_Switches_{"SplitDie" if split_die else "Normal"}'
            ))
        
        # 2. Sub-plane交换机
        # 只为实际存在连接的Sub-plane创建交换机
        for sp_id in range(self.constants.NUM_SUBPLANES):
            connected_npus = []
            
            for node_id in slice_nodes:
                if node_id in self.router.subplane_connectivity[sp_id]:
                    connected_npus.extend(node_to_npus[node_id])
            
            if len(connected_npus) > 1:
                # 该Sub-plane的总带宽 - 这是物理固定值
                subplane_bw = (self.constants.SWITCHES_PER_SUBPLANE * 
                              self.constants.SUBPLANE_BW_PER_SWITCH)
                
                if split_die:
                    subplane_bw *= self.constants.SPLIT_DIE_BW_FACTOR
                
                switches.append((
                    connected_npus,
                    connected_npus,
                    subplane_bw,
                    f'SubPlane{sp_id}_L2_Switches'
                ))
        
        # 3. 顶层互联 - 固定的系统级带宽
        if len(slice_nodes) > 1:
            all_npus = []
            for node_npus in node_to_npus.values():
                all_npus.extend(node_npus)
            
            # 这是整个CM384系统的固定带宽，不会因为切片而改变
            fabric_bw = self.constants.TOTAL_INTER_FABRIC_BW
            if split_die:
                fabric_bw *= self.constants.SPLIT_DIE_BW_FACTOR
            
            switches.append((
                all_npus,
                all_npus,
                fabric_bw,
                'CM384_L3_InterFabric_FixedBW'
            ))
        
        return switches

class CM384LogicalTopologyMapper:
    """CM384逻辑拓扑映射器 - 在物理拓扑上实现逻辑拓扑"""
    
    def __init__(self, physical_topology: Topology, slice_nodes: List[int], 
                 split_die: bool = False):
        self.physical_topo = physical_topology
        self.slice_nodes = slice_nodes
        self.split_die = split_die
        self.num_npus = physical_topology.num_nodes()
        self.npus_per_node = 16 if split_die else 8
        
    def create_ring_topology(self, ring_size: Optional[int] = None) -> List[int]:
        """
        创建优化的ring拓扑映射
        
        策略：
        1. 优先利用节点内高带宽连接
        2. 节点间连接选择共享Sub-plane最多的路径
        3. 考虑拓扑的异构性
        """
        if ring_size is None:
            ring_size = self.num_npus
        
        if ring_size > self.num_npus:
            raise ValueError(f"Ring大小超过可用NPU数量")
        
        # 获取节点分组
        node_groups = []
        for i in range(len(self.slice_nodes)):
            start_npu = i * self.npus_per_node
            end_npu = start_npu + self.npus_per_node
            node_groups.append(list(range(start_npu, min(end_npu, self.num_npus))))
        
        ring = []
        
        # 策略1：如果ring_size <= 节点数，每个节点贡献一个NPU
        if ring_size <= len(node_groups):
            for i in range(ring_size):
                # 选择每个节点内带宽最优的NPU（通常是第一个）
                ring.append(node_groups[i][0])
        else:
            # 策略2：需要多个节点内NPU
            # 首先每个节点贡献一个NPU
            for node_group in node_groups:
                ring.append(node_group[0])
            
            # 然后在需要的节点内添加更多NPU
            remaining = ring_size - len(node_groups)
            for i in range(remaining):
                node_idx = i % len(node_groups)
                if len(node_groups[node_idx]) > 1:
                    npu_idx = 1 + (i // len(node_groups))
                    if npu_idx < len(node_groups[node_idx]):
                        ring.append(node_groups[node_idx][npu_idx])
        
        return ring[:ring_size]
    
    def create_mesh_topology(self, mesh_dims: Tuple[int, int]) -> List[List[int]]:
        """
        创建2D mesh拓扑映射
        
        策略：
        1. 同一行的NPU尽量在同一节点内
        2. 列间连接优化节点间带宽
        """
        rows, cols = mesh_dims
        if rows * cols > self.num_npus:
            raise ValueError(f"Mesh大小超过可用NPU数量")
        
        mesh = []
        npu_idx = 0
        
        # 获取节点分组
        node_groups = []
        for i in range(len(self.slice_nodes)):
            start_npu = i * self.npus_per_node
            end_npu = start_npu + self.npus_per_node
            node_groups.append(list(range(start_npu, min(end_npu, self.num_npus))))
        
        # 分配策略：每行尽量在同一节点内
        for row in range(rows):
            mesh_row = []
            for col in range(cols):
                if npu_idx < self.num_npus:
                    # 计算应该使用哪个节点
                    preferred_node = (row * cols + col) // self.npus_per_node
                    if preferred_node < len(node_groups):
                        local_idx = (row * cols + col) % self.npus_per_node
                        if local_idx < len(node_groups[preferred_node]):
                            mesh_row.append(node_groups[preferred_node][local_idx])
                        else:
                            mesh_row.append(npu_idx)
                    else:
                        mesh_row.append(npu_idx)
                    npu_idx += 1
                else:
                    mesh_row.append(-1)
            mesh.append(mesh_row)
        
        return mesh
    
    def create_star_topology(self, center_npu: Optional[int] = None) -> Tuple[int, List[int]]:
        """
        创建star拓扑映射
        
        策略：
        1. 选择位于拓扑中心的NPU作为中心
        2. 叶子节点按带宽优先级排序
        """
        if center_npu is None:
            # 选择第一个节点的第一个NPU作为中心
            center_npu = 0
        
        if center_npu >= self.num_npus:
            raise ValueError(f"中心NPU ID超出范围")
        
        # 获取所有其他NPU
        all_npus = list(range(self.num_npus))
        all_npus.remove(center_npu)
        
        # 按照与中心NPU的带宽排序
        def get_bandwidth_to_center(npu_id):
            return self.physical_topo.links[npu_id][center_npu]
        
        # 排序叶子节点：带宽高的优先
        leaf_npus = sorted(all_npus, key=get_bandwidth_to_center, reverse=True)
        
        return center_npu, leaf_npus
    
    def create_tree_topology(self, tree_radix: int = 2) -> Dict[str, List[int]]:
        """
        创建树拓扑映射
        
        策略：
        1. 根节点选择拓扑中心
        2. 子节点按带宽和位置优化分配
        """
        if tree_radix < 2:
            raise ValueError("树的分支因子必须至少为2")
        
        # 选择根节点（第一个节点的第一个NPU）
        root = 0
        
        # 构建树层次
        tree_levels = []
        remaining_npus = list(range(1, self.num_npus))
        
        current_level = [root]
        tree_levels.append(current_level)
        
        while remaining_npus:
            next_level = []
            
            for parent in current_level:
                # 为每个父节点分配子节点
                for _ in range(tree_radix):
                    if remaining_npus:
                        # 选择与父节点带宽最高的NPU
                        best_child = max(remaining_npus, 
                                       key=lambda npu: self.physical_topo.links[npu][parent])
                        next_level.append(best_child)
                        remaining_npus.remove(best_child)
            
            if next_level:
                tree_levels.append(next_level)
            current_level = next_level
        
        return {
            'root': [root],
            'levels': tree_levels,
            'total_levels': len(tree_levels)
        }

# 全局实例
_cm384_constants = CM384PhysicalConstants()
_cm384_builder = CM384TopologyBuilder(_cm384_constants)

def create_cm384_slice(slice_nodes: List[int], split_die: bool = False) -> Topology:
    """
    创建CM384切片拓扑
    
    Args:
        slice_nodes: 切片包含的节点列表 (0-47)
        split_die: 是否启用910C分die模式
    
    Returns:
        Topology: 真实的CM384切片拓扑
    """
    return _cm384_builder.build_slice_topology(slice_nodes, split_die)

def cm384(num_npus: int = 128, split_die: bool = False) -> Topology:
    """
    创建标准CM384拓扑切片
    
    Args:
        num_npus: NPU数量
        split_die: 是否启用分die模式
    
    Returns:
        Topology: CM384拓扑
    """
    npus_per_node = 16 if split_die else 8
    if num_npus % npus_per_node != 0:
        raise ValueError(f"NPU数量必须是{npus_per_node}的倍数")
    
    num_nodes = num_npus // npus_per_node
    if num_nodes > _cm384_constants.TOTAL_NODES:
        raise ValueError(f"节点数量不能超过{_cm384_constants.TOTAL_NODES}")
    
    # 选择连续的节点作为切片
    slice_nodes = list(range(num_nodes))
    return create_cm384_slice(slice_nodes, split_die)

def cm384_full(split_die: bool = False) -> Topology:
    """创建完整的CM384拓扑"""
    all_nodes = list(range(_cm384_constants.TOTAL_NODES))
    return create_cm384_slice(all_nodes, split_die)

def cm384_256npu() -> Topology:
    """创建256-NPU CM384拓扑 (16节点，分die模式)"""
    return cm384(256, split_die=True)

def get_cm384_logical_mapper(physical_topology: Topology, slice_nodes: List[int], 
                            split_die: bool = False) -> CM384LogicalTopologyMapper:
    """
    获取CM384逻辑拓扑映射器
    
    Args:
        physical_topology: 物理拓扑
        slice_nodes: 切片节点列表
        split_die: 是否分die模式
    
    Returns:
        CM384LogicalTopologyMapper: 逻辑拓扑映射器
    """
    return CM384LogicalTopologyMapper(physical_topology, slice_nodes, split_die)

# 便捷函数
def create_cm384_ring(num_npus: int = 128, split_die: bool = False) -> Tuple[Topology, List[int]]:
    """创建CM384 ring拓扑"""
    topo = cm384(num_npus, split_die)
    npus_per_node = 16 if split_die else 8
    slice_nodes = list(range(num_npus // npus_per_node))
    mapper = get_cm384_logical_mapper(topo, slice_nodes, split_die)
    ring = mapper.create_ring_topology()
    return topo, ring

def create_cm384_mesh(mesh_dims: Tuple[int, int], split_die: bool = False) -> Tuple[Topology, List[List[int]]]:
    """创建CM384 mesh拓扑"""
    num_npus = mesh_dims[0] * mesh_dims[1]
    npus_per_node = 16 if split_die else 8
    # 向上取整到npus_per_node的倍数
    num_npus = ((num_npus + npus_per_node - 1) // npus_per_node) * npus_per_node
    
    topo = cm384(num_npus, split_die)
    slice_nodes = list(range(num_npus // npus_per_node))
    mapper = get_cm384_logical_mapper(topo, slice_nodes, split_die)
    mesh = mapper.create_mesh_topology(mesh_dims)
    return topo, mesh

def create_cm384_star(num_npus: int = 128, split_die: bool = False) -> Tuple[Topology, int, List[int]]:
    """创建CM384 star拓扑"""
    topo = cm384(num_npus, split_die)
    npus_per_node = 16 if split_die else 8
    slice_nodes = list(range(num_npus // npus_per_node))
    mapper = get_cm384_logical_mapper(topo, slice_nodes, split_die)
    center, leaves = mapper.create_star_topology()
    return topo, center, leaves