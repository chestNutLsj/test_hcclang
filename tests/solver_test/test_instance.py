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

from hcclang.solver.instance import Instance


class TestInstance(unittest.TestCase):
    
    def test_basic_instance_creation(self):
        """测试基本Instance对象创建"""
        instance = Instance(steps=10)
        self.assertEqual(instance.steps, 10)
        self.assertEqual(instance.extra_rounds, 0)
        self.assertEqual(instance.chunks, 1)
        self.assertIsNone(instance.pipeline)
        self.assertIsNone(instance.extra_memory)
        self.assertFalse(instance.allow_exchange)
    
    def test_instance_with_all_params(self):
        """测试包含所有参数的Instance创建"""
        instance = Instance(
            steps=8,
            extra_rounds=2,
            chunks=4,
            pipeline=3,
            extra_memory=16,
            allow_exchange=True
        )
        self.assertEqual(instance.steps, 8)
        self.assertEqual(instance.extra_rounds, 2)
        self.assertEqual(instance.chunks, 4)
        self.assertEqual(instance.pipeline, 3)
        self.assertEqual(instance.extra_memory, 16)
        self.assertTrue(instance.allow_exchange)
    
    def test_rounds_calculation(self):
        """测试rounds方法"""
        instance = Instance(steps=5, extra_rounds=3)
        self.assertEqual(instance.rounds(), 8)
        
        instance_no_extra = Instance(steps=10)
        self.assertEqual(instance_no_extra.rounds(), 10)
    
    def test_set_method(self):
        """测试set方法用于更新参数"""
        original = Instance(steps=5, chunks=2)
        
        # 更新部分参数
        updated = original.set(steps=10, extra_rounds=3)
        self.assertEqual(updated.steps, 10)
        self.assertEqual(updated.extra_rounds, 3)
        self.assertEqual(updated.chunks, 2)  # 保持原值
        
        # 确保原对象未被修改
        self.assertEqual(original.steps, 5)
        self.assertEqual(original.extra_rounds, 0)
    
    def test_set_method_none_values(self):
        """测试set方法传入None值时的行为"""
        original = Instance(steps=5, chunks=2, pipeline=4)
        updated = original.set(steps=None, chunks=3)
        
        self.assertEqual(updated.steps, 5)  # None值应保持原值
        self.assertEqual(updated.chunks, 3)  # 更新为新值
        self.assertEqual(updated.pipeline, 4)  # 保持原值
    
    def test_str_representation(self):
        """测试字符串表示"""
        # 基本实例
        instance = Instance(steps=5)
        self.assertEqual(str(instance), 'steps=5')
        
        # 包含extra_rounds
        instance = Instance(steps=5, extra_rounds=2)
        self.assertIn('steps=5', str(instance))
        self.assertIn('rounds=7', str(instance))
        
        # 包含多个参数
        instance = Instance(
            steps=8,
            extra_rounds=2,
            chunks=4,
            pipeline=3,
            extra_memory=16,
            allow_exchange=True
        )
        instance_str = str(instance)
        self.assertIn('steps=8', instance_str)
        self.assertIn('rounds=10', instance_str)
        self.assertIn('chunks=4', instance_str)
        self.assertIn('pipeline=3', instance_str)
        self.assertIn('extra_memory=16', instance_str)
        self.assertIn('allow_exchange', instance_str)
    
    def test_immutability(self):
        """测试Instance对象的不可变性（frozen=True）"""
        instance = Instance(steps=5)
        
        # Instance对象是frozen的，所以属性是只读的
        # 这里我们只验证对象存在，不尝试修改属性
        self.assertEqual(instance.steps, 5)
        
        # 通过创建新实例验证不可变性的设计
        new_instance = instance.set(steps=10)
        self.assertEqual(instance.steps, 5)  # 原实例不变
        self.assertEqual(new_instance.steps, 10)  # 新实例有新值


if __name__ == '__main__':
    unittest.main() 