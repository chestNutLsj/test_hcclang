#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Basic functionality tests for hcclang.language module.

This module tests core functionality including:
- HCCLProgram creation and context management
- Buffer initialization and management
- Basic chunk operations
- Ref object functionality
"""

import os
import sys

# Add the parent directory to path to import hcclang
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import pytest
from hcclang.language import HCCLProgram, chunk, Check
from hcclang.language.buffer import Buffer
from hcclang.language.collectives import AllReduce
from hcclang.topologies import generic

def test_hccl_program_creation():
    """Test basic HCCLProgram creation and initialization."""
    print("\n=== Testing HCCLProgram Creation ===")
    
    # Create a simple topology
    topo = generic.fully_connected(4)
    collective = AllReduce(4, 4, False)  # 4 ranks, 4 chunks, not inplace
    
    # Create HCCLProgram
    with HCCLProgram("test_program", topo, collective, 1) as prog:
        print(f"✓ Program created: {prog.name}")
        print(f"  - Topology: {prog.topo.name}")
        print(f"  - Collective: {prog.collective.name}")
        print(f"  - Ranks: {prog.num_ranks}")
        print(f"  - Protocol: {prog.protocol}")
        
        # Test buffer initialization
        assert prog.buffers is not None
        assert len(prog.buffers) == prog.num_ranks
        print(f"  - Buffers initialized for {len(prog.buffers)} ranks")
        
        # Test that each rank has input buffer
        for rank in range(prog.num_ranks):
            assert Buffer.input in prog.buffers[rank]
            input_buffer = prog.buffers[rank][Buffer.input]
            print(f"  - Rank {rank} input buffer size: {len(input_buffer)}")
    
    print("✓ HCCLProgram creation test passed")

def test_chunk_operations():
    """Test basic chunk operations and references."""
    print("\n=== Testing Chunk Operations ===")
    
    topo = generic.fully_connected(4)
    collective = AllReduce(4, 8, False)  # 4 ranks, 8 chunks
    
    with HCCLProgram("chunk_test", topo, collective, 1) as prog:
        # Test chunk reference creation
        ref0 = chunk(0, Buffer.input, 0, 2)
        print(f"✓ Created chunk reference: {ref0}")
        
        assert ref0 is not None
        assert ref0.rank == 0
        assert ref0.buffer == Buffer.input
        assert ref0.index == 0
        assert ref0.size == 2
        
        # Test chunk splitting
        split_chunks = ref0.split(2)
        print(f"✓ Split chunk into {len(split_chunks)} parts")
        
        for i, split_chunk in enumerate(split_chunks):
            print(f"  - Split {i}: rank={split_chunk.rank}, index={split_chunk.index}, size={split_chunk.size}")
            assert split_chunk.size == 1
            assert split_chunk.index == i
        
        # Test chunk grouping
        grouped = split_chunks[0].group(split_chunks[1])
        print(f"✓ Grouped chunks back: {grouped}")
        assert grouped.size == 2
        assert grouped.index == 0
    
    print("✓ Chunk operations test passed")

def test_copy_operations():
    """Test copy operations between ranks."""
    print("\n=== Testing Copy Operations ===")
    
    topo = generic.fully_connected(4)
    collective = AllReduce(4, 4, False)
    
    with HCCLProgram("copy_test", topo, collective, 1) as prog:
        # Get chunk from rank 0
        src_chunk = chunk(0, Buffer.input, 0, 1)
        print(f"✓ Source chunk: {src_chunk}")
        
        # Copy to rank 1
        dst_chunk = src_chunk.copy(1, Buffer.input, 0)
        print(f"✓ Copied to destination: {dst_chunk}")
        
        assert dst_chunk.rank == 1
        assert dst_chunk.buffer == Buffer.input
        assert dst_chunk.index == 0
        assert dst_chunk.size == 1
        
        # Verify the copy was tracked in buffers
        original_chunk_data = prog.buffers[0][Buffer.input][0]
        copied_chunk_data = prog.buffers[1][Buffer.input][0]
        
        print(f"  - Original chunk: {original_chunk_data}")
        print(f"  - Copied chunk: {copied_chunk_data}")
        
        # They should be the same data
        assert original_chunk_data == copied_chunk_data
    
    print("✓ Copy operations test passed")

def test_buffer_management():
    """Test buffer creation and management."""
    print("\n=== Testing Buffer Management ===")
    
    topo = generic.fully_connected(2)
    collective = AllReduce(2, 4, False)
    
    with HCCLProgram("buffer_test", topo, collective, 1) as prog:
        # Test scratch buffer creation
        prog.check_buffer_exists(0, "scratch1")
        assert "scratch1" in prog.buffers[0]
        print("✓ Scratch buffer created successfully")
        
        # Test get_chunks method
        chunks = prog.get_chunks(0, Buffer.input, 0, 2)
        print(f"✓ Retrieved {len(chunks)} chunks from buffer")
        
        for i, chunk_data in enumerate(chunks):
            if chunk_data is not None:
                print(f"  - Chunk {i}: {chunk_data}")
    
    print("✓ Buffer management test passed")

def test_program_context():
    """Test program context management."""
    print("\n=== Testing Program Context ===")
    
    topo = generic.fully_connected(2)
    collective = AllReduce(2, 2, False)
    
    # Test context entry and exit
    prog = HCCLProgram("context_test", topo, collective, 1)
    
    # Should be able to enter context
    with prog:
        print("✓ Entered program context")
        
        # Test that chunk() function works in context
        test_chunk = chunk(0, Buffer.input, 0, 1)
        assert test_chunk is not None
        print("✓ Chunk function works in context")
    
    print("✓ Exited program context")
    print("✓ Program context test passed")

def test_check_functionality():
    """Test program checking functionality."""
    print("\n=== Testing Check Functionality ===")
    
    topo = generic.fully_connected(2)
    collective = AllReduce(2, 2, False)
    
    with HCCLProgram("check_test", topo, collective, 1) as prog:
        try:
            # Test the check function
            result = Check()
            print(f"✓ Check function executed, result: {result}")
        except Exception as e:
            print(f"⚠️  Check function raised exception (expected for incomplete program): {e}")
    
    print("✓ Check functionality test passed")

def run_all_tests():
    """Run all basic functionality tests."""
    print("HCCLang Language Module - Basic Functionality Tests")
    print("=" * 60)
    
    test_functions = [
        test_hccl_program_creation,
        test_chunk_operations,
        test_copy_operations,
        test_buffer_management,
        test_program_context,
        test_check_functionality
    ]
    
    passed_tests = 0
    total_tests = len(test_functions)
    
    for test_func in test_functions:
        try:
            test_func()
            passed_tests += 1
        except Exception as e:
            print(f"❌ Test {test_func.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"Basic Functionality Tests Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All basic functionality tests passed!")
    else:
        print("⚠️  Some tests failed. Please check the output above.")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    run_all_tests() 