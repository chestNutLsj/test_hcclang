# SPDX-License-Identifier: GPL-2.0-only

#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Import and basic tests for hcclang.language module.
This ensures all modules can be imported correctly.
"""

import os
import sys

# Add the parent directory to path to import hcclang
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

def test_imports():
    """Test that all language modules can be imported."""
    print("=== Testing Module Imports ===")
    
    try:
        # Test core language imports
        from hcclang.language import HCCLProgram, chunk, Check, XML
        print("✓ Core language module imported successfully")
        
        # Test buffer module
        from hcclang.language.buffer import Buffer
        print("✓ Buffer module imported successfully")
        
        # Test collectives module
        from hcclang.language.collectives import AllReduce, Broadcast, ReduceScatter
        print("✓ Collectives module imported successfully")
        
        # Test IR module
        from hcclang.language.ir import Program, Instruction
        print("✓ IR module imported successfully")
        
        # Test rank DAG module
        from hcclang.language.rank_dag import InstructionDAG
        print("✓ Rank DAG module imported successfully")
        
        # Test topology imports
        from hcclang.topologies import generic
        print("✓ Topologies module imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_basic_topology_creation():
    """Test basic topology creation."""
    print("\n=== Testing Basic Topology Creation ===")
    
    try:
        from hcclang.topologies import generic
        
        # Test simple topologies
        ring_topo = generic.ring(4)
        print(f"✓ Ring topology created: {ring_topo.name}, {ring_topo.num_nodes()} nodes")
        
        fc_topo = generic.fully_connected(3)
        print(f"✓ Fully connected topology created: {fc_topo.name}, {fc_topo.num_nodes()} nodes")
        
        return True
        
    except Exception as e:
        print(f"❌ Topology creation failed: {e}")
        return False

def test_basic_collective_creation():
    """Test basic collective creation."""
    print("\n=== Testing Basic Collective Creation ===")
    
    try:
        from hcclang.language.collectives import AllReduce, Broadcast
        
        # Test AllReduce creation
        allreduce = AllReduce(4, 8, False)  # 4 ranks, 8 chunks, not inplace
        print(f"✓ AllReduce collective created: {allreduce.name}")
        print(f"  - Ranks: {allreduce.num_ranks}")
        print(f"  - Chunks: {allreduce.num_chunks}")
        print(f"  - Inplace: {allreduce.inplace}")
        
        # Test Broadcast creation
        broadcast = Broadcast(4, 8, 0)  # 4 ranks, 8 chunks, root=0
        print(f"✓ Broadcast collective created: {broadcast.name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Collective creation failed: {e}")
        return False

def test_simple_program_creation():
    """Test creating a simple HCCLProgram."""
    print("\n=== Testing Simple Program Creation ===")
    
    try:
        from hcclang.language import HCCLProgram
        from hcclang.language.collectives import AllReduce
        from hcclang.topologies import generic
        
        # Create components
        topo = generic.ring(4)
        collective = AllReduce(4, 4, False)
        
        # Create program
        prog = HCCLProgram("test_program", topo, collective, 1)
        print(f"✓ HCCLProgram created successfully")
        print(f"  - Name: {prog.name}")
        print(f"  - Topology: {prog.topo.name}")
        print(f"  - Collective: {prog.collective.name}")
        print(f"  - Ranks: {prog.num_ranks}")
        
        return True
        
    except Exception as e:
        print(f"❌ Program creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_buffer_enum():
    """Test Buffer enum functionality."""
    print("\n=== Testing Buffer Enum ===")
    
    try:
        from hcclang.language.buffer import Buffer
        
        print(f"✓ Buffer.input: {Buffer.input}")
        print(f"✓ Buffer.output: {Buffer.output}")
        print(f"✓ Buffer.scratch: {Buffer.scratch}")
        
        return True
        
    except Exception as e:
        print(f"❌ Buffer enum test failed: {e}")
        return False

def run_all_tests():
    """Run all import and basic tests."""
    print("HCCLang Language Module - Import and Basic Tests")
    print("=" * 60)
    
    test_functions = [
        test_imports,
        test_basic_topology_creation,
        test_basic_collective_creation,
        test_buffer_enum,
        test_simple_program_creation,
    ]
    
    passed_tests = 0
    total_tests = len(test_functions)
    
    for test_func in test_functions:
        try:
            if test_func():
                passed_tests += 1
            else:
                print(f"❌ Test {test_func.__name__} failed")
        except Exception as e:
            print(f"❌ Test {test_func.__name__} failed with exception: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"Import and Basic Tests Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All import and basic tests passed!")
        return True
    else:
        print("⚠️  Some tests failed. Please check the output above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 