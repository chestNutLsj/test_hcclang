#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
import os
import unittest

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from hcclang.solver.isomorphisms import find_isomorphisms, Permutation
from hcclang.topologies import generic


class TestIsomorphisms(unittest.TestCase):
    
    def test_permutation_creation(self):
        """测试Permutation对象创建"""
        perm = Permutation([1, 0, 2])
        self.assertEqual(perm.nodes, [1, 0, 2])
        self.assertIn('Permutation(nodes=[1, 0, 2])', str(perm))
    
    def test_identity_isomorphism(self):
        """测试相同拓扑的恒等同构"""
        topology = generic.fully_connected(4)
        target_topology = generic.fully_connected(4)
        
        isomorphisms = find_isomorphisms(topology, target_topology, limit=1)
        
        # 至少应该找到一个恒等映射
        self.assertGreater(len(isomorphisms), 0)
        
        # 检查返回的是Permutation对象
        self.assertIsInstance(isomorphisms[0], Permutation)
        self.assertEqual(len(isomorphisms[0].nodes), 4)
    
    def test_different_size_topologies(self):
        """测试不同大小拓扑的同构查找应该报错"""
        topology1 = generic.fully_connected(4)
        topology2 = generic.fully_connected(6)
        
        with self.assertRaises(ValueError) as context:
            find_isomorphisms(topology1, topology2)
        
        self.assertIn('HCCL error', str(context.exception))
        self.assertIn('target topology does not match', str(context.exception))
    
    def test_limit_parameter(self):
        """测试limit参数的功能"""
        topology = generic.fully_connected(4)
        target_topology = generic.fully_connected(4)
        
        # 测试limit=1
        isomorphisms_1 = find_isomorphisms(topology, target_topology, limit=1)
        self.assertEqual(len(isomorphisms_1), 1)
        
        # 测试limit=3
        isomorphisms_3 = find_isomorphisms(topology, target_topology, limit=3)
        self.assertLessEqual(len(isomorphisms_3), 3)
    
    def test_invalid_limit(self):
        """测试无效的limit参数"""
        topology = generic.fully_connected(4)
        target_topology = generic.fully_connected(4)
        
        with self.assertRaises(ValueError) as context:
            find_isomorphisms(topology, target_topology, limit=0)
        
        self.assertIn('HCCL error', str(context.exception))
        self.assertIn('limit was set improperly', str(context.exception))
        
        with self.assertRaises(ValueError) as context:
            find_isomorphisms(topology, target_topology, limit=-1)
        
        self.assertIn('HCCL error', str(context.exception))
    
    def test_line_topology_isomorphisms(self):
        """测试线性拓扑的同构"""
        topology = generic.line(4)
        target_topology = generic.line(4)
        
        isomorphisms = find_isomorphisms(topology, target_topology, limit=10)
        
        # 线性拓扑应该有2个同构映射（正向和反向）
        self.assertGreaterEqual(len(isomorphisms), 1)
        self.assertLessEqual(len(isomorphisms), 2)
        
        for perm in isomorphisms:
            self.assertIsInstance(perm, Permutation)
            self.assertEqual(len(perm.nodes), 4)
    
    def test_logging_parameter(self):
        """测试logging参数不会引发错误"""
        topology = generic.fully_connected(3)
        target_topology = generic.fully_connected(3)
        
        # 这应该不会抛出异常，即使打印信息
        isomorphisms = find_isomorphisms(topology, target_topology, limit=1, logging=True)
        self.assertGreater(len(isomorphisms), 0)


if __name__ == '__main__':
    unittest.main() 