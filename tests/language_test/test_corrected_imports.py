# SPDX-License-Identifier: GPL-2.0-only

#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Corrected import and basic tests for hcclang.language module.
This uses the correct import paths based on actual module structure.
"""

import os
import sys

# Add the parent directory to path to import hcclang
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

def test_imports():
    """Test that all language modules can be imported with correct paths."""
    print("=== Testing Module Imports (Corrected) ===")
    
    try:
        # Test core language imports
        from hcclang.language import HCCLProgram, chunk, Check, XML
        print("✓ Core language module imported successfully")
        
        # Test buffer module (BufferSlice class)
        from hcclang.language.buffer import BufferSlice
        print("✓ BufferSlice from buffer module imported successfully")
        
        # Test Buffer enum from ir module (this is the correct location)
        from hcclang.language.ir import Buffer, Instruction, Program
        print("✓ Buffer enum from ir module imported successfully")
        
        # Test collectives module (available classes)
        from hcclang.language.collectives import AllReduce, AllGather, ReduceScatter, AllToAll
        print("✓ Collectives module imported successfully")
        
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

def test_buffer_enum():
    """Test Buffer enum functionality from ir module."""
    print("\n=== Testing Buffer Enum (from ir module) ===")
    
    try:
        from hcclang.language.ir import Buffer
        
        print(f"✓ Buffer.input: {Buffer.input}")
        print(f"✓ Buffer.output: {Buffer.output}")
        print(f"✓ Buffer.scratch: {Buffer.scratch}")
        print(f"✓ Buffer enum string representation works: {str(Buffer.input)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Buffer enum test failed: {e}")
        return False

def test_available_collectives():
    """Test creation of available collective types."""
    print("\n=== Testing Available Collectives ===")
    
    try:
        from hcclang.language.collectives import AllReduce, AllGather, ReduceScatter, AllToAll
        
        # Test AllReduce creation
        allreduce = AllReduce(4, 8, False)  # 4 ranks, 8 chunks, not inplace
        print(f"✓ AllReduce collective created: {allreduce.name}")
        print(f"  - Ranks: {allreduce.num_ranks}")
        print(f"  - Chunk factor: {allreduce.chunk_factor}")
        print(f"  - Inplace: {allreduce.inplace}")
        
        # Test AllGather creation
        allgather = AllGather(4, 8, False)  # 4 ranks, 8 chunks, not inplace
        print(f"✓ AllGather collective created: {allgather.name}")
        
        # Test ReduceScatter creation
        reducescatter = ReduceScatter(4, 2, False)
        print(f"✓ ReduceScatter collective created: {reducescatter.name}")
        
        # Test AllToAll creation
        alltoall = AllToAll(4, 2, False)
        print(f"✓ AllToAll collective created: {alltoall.name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Collective creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_program_with_context():
    """Test creating and using HCCLProgram with context manager."""
    print("\n=== Testing Program with Context Manager ===")
    
    try:
        from hcclang.language import HCCLProgram, chunk
        from hcclang.language.collectives import AllReduce
        from hcclang.language.ir import Buffer
        from hcclang.topologies import generic
        
        # Create components
        topo = generic.ring(4)
        collective = AllReduce(4, 4, False)
        
        # Test context manager usage
        with HCCLProgram("test_program", topo, collective, 1) as prog:
            print(f"✓ HCCLProgram context entered successfully")
            print(f"  - Name: {prog.name}")
            print(f"  - Topology: {prog.topo.name}")
            print(f"  - Collective: {prog.collective.name}")
            print(f"  - Ranks: {prog.num_ranks}")
            
            # Test chunk creation within context
            test_chunk = chunk(0, Buffer.input, 0, 1)
            print(f"✓ Chunk created in context: {test_chunk}")
            
            # Test buffer access
            buffers = prog.buffers
            print(f"✓ Buffers accessible: {len(buffers)} rank buffers")
            
            for rank in range(prog.num_ranks):
                input_buf = buffers[rank][Buffer.input]
                print(f"  - Rank {rank} input buffer size: {len(input_buf)}")
        
        print("✓ Program context exited successfully")
        return True
        
    except Exception as e:
        print(f"❌ Program context test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_chunk_operations():
    """Test chunk operations within a program context."""
    print("\n=== Testing Chunk Operations ===")
    
    try:
        from hcclang.language import HCCLProgram, chunk
        from hcclang.language.collectives import AllReduce
        from hcclang.language.ir import Buffer
        from hcclang.topologies import generic
        
        topo = generic.fully_connected(4)
        collective = AllReduce(4, 8, False)  # 4 ranks, 8 chunks
        
        with HCCLProgram("chunk_test", topo, collective, 1) as prog:
            # Test chunk reference creation
            ref0 = chunk(0, Buffer.input, 0, 2)
            print(f"✓ Created chunk reference: {ref0}")
            
            # Test chunk properties
            print(f"  - Rank: {ref0.rank}")
            print(f"  - Buffer: {ref0.buffer}")
            print(f"  - Index: {ref0.index}")
            print(f"  - Size: {ref0.size}")
            
            # Test chunk splitting
            split_chunks = ref0.split(2)
            print(f"✓ Split chunk into {len(split_chunks)} parts")
            
            for i, split_chunk in enumerate(split_chunks):
                print(f"  - Split {i}: rank={split_chunk.rank}, index={split_chunk.index}, size={split_chunk.size}")
            
            # Test chunk grouping
            grouped = split_chunks[0].group(split_chunks[1])
            print(f"✓ Grouped chunks back: {grouped}")
            print(f"  - Grouped size: {grouped.size}")
            print(f"  - Grouped index: {grouped.index}")
        
        return True
        
    except Exception as e:
        print(f"❌ Chunk operations test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_instruction_dag():
    """Test InstructionDAG functionality."""
    print("\n=== Testing InstructionDAG ===")
    
    try:
        from hcclang.language.rank_dag import InstructionDAG
        from hcclang.language.ir import Buffer
        
        # Create a simple instruction DAG
        num_ranks = 2
        buffers = {}
        for r in range(num_ranks):
            buffers[r] = {Buffer.input: [None] * 4}
        
        dag = InstructionDAG(num_ranks, buffers)
        print(f"✓ InstructionDAG created for {num_ranks} ranks")
        print(f"  - Number of ranks: {dag.num_ranks}")
        print(f"  - Operations dict: {type(dag.operations)}")
        print(f"  - Threadblocks: {len(dag.tbs)} rank TBs initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ InstructionDAG test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all corrected import and basic tests."""
    print("HCCLang Language Module - Corrected Import and Basic Tests")
    print("=" * 70)
    
    test_functions = [
        test_imports,
        test_buffer_enum,
        test_available_collectives,
        test_program_with_context,
        test_chunk_operations,
        test_instruction_dag,
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
    
    print("\n" + "=" * 70)
    print(f"Corrected Tests Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All corrected tests passed!")
        return True
    else:
        print("⚠️  Some tests failed. Please check the output above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 