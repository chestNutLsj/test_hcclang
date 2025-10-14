# SPDX-License-Identifier: GPL-2.0-only

#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
import os
import unittest

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)


class TestSolverIntegration(unittest.TestCase):
    
    def test_solver_imports(self):
        """测试solver模块的所有组件能正常导入"""
        try:
            from hcclang.solver.instance import Instance
            from hcclang.solver.isomorphisms import find_isomorphisms, Permutation
            from hcclang.solver.steps_bound import lower_bound_steps
            from hcclang.solver.rounds_bound import lower_bound_rounds
            from hcclang.solver.path_encoding import PathEncodingBase
            self.assertTrue(True, "All solver modules imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import solver modules: {e}")
    
    def test_instance_basic_functionality(self):
        """测试Instance类的基本功能"""
        from hcclang.solver.instance import Instance
        
        # 创建实例
        instance = Instance(steps=5, chunks=2)
        self.assertEqual(instance.steps, 5)
        self.assertEqual(instance.chunks, 2)
        
        # 测试rounds方法
        self.assertEqual(instance.rounds(), 5)
        
        # 测试set方法
        new_instance = instance.set(steps=10)
        self.assertEqual(new_instance.steps, 10)
        self.assertEqual(new_instance.chunks, 2)  # 保持原值
    
    def test_permutation_basic_functionality(self):
        """测试Permutation类的基本功能"""
        from hcclang.solver.isomorphisms import Permutation
        
        perm = Permutation([1, 0, 2])
        self.assertEqual(perm.nodes, [1, 0, 2])
        self.assertIn('Permutation', str(perm))
    
    def test_topology_creation(self):
        """测试拓扑创建能正常工作"""
        try:
            from hcclang.topologies import generic
            
            # 创建基本拓扑
            fc_topo = generic.fully_connected(4)
            self.assertEqual(fc_topo.num_nodes(), 4)
            
            line_topo = generic.line(3)
            self.assertEqual(line_topo.num_nodes(), 3)
            
            self.assertTrue(True, "Topology creation works")
        except Exception as e:
            self.fail(f"Topology creation failed: {e}")
    
    def test_collective_creation(self):
        """测试collective创建能正常工作"""
        try:
            from hcclang.language.collectives import AllReduce, AllGather
            
            # 创建AllReduce collective
            allreduce = AllReduce(4, 1, False)
            self.assertEqual(allreduce.num_ranks, 4)
            self.assertEqual(allreduce.chunk_factor, 1)
            self.assertFalse(allreduce.inplace)
            
            # 创建AllGather collective
            allgather = AllGather(3, 2, True)
            self.assertEqual(allgather.num_ranks, 3)
            self.assertEqual(allgather.chunk_factor, 2)
            self.assertTrue(allgather.inplace)
            
            self.assertTrue(True, "Collective creation works")
        except Exception as e:
            self.fail(f"Collective creation failed: {e}")
    
    def test_solver_components_exist(self):
        """测试solver组件都存在并且可以实例化"""
        try:
            from hcclang.solver.instance import Instance
            from hcclang.solver.isomorphisms import Permutation
            from hcclang.topologies import generic
            from hcclang.language.collectives import AllReduce
            
            # 创建所需对象
            instance = Instance(steps=3)
            perm = Permutation([0, 1, 2])
            topology = generic.fully_connected(3)
            collective = AllReduce(3, 1, False)
            
            # 验证对象创建成功
            self.assertIsNotNone(instance)
            self.assertIsNotNone(perm)
            self.assertIsNotNone(topology)
            self.assertIsNotNone(collective)
            
        except Exception as e:
            self.fail(f"Solver component instantiation failed: {e}")


if __name__ == '__main__':
    unittest.main() 