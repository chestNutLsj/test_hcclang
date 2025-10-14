# SPDX-License-Identifier: GPL-2.0-only

#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Advanced functionality tests for hcclang.language module.

This module tests more complex functionality including:
- Actual collective communication operations
- XML generation and IR lowering
- Copy and reduce operations
- Program validation
"""

import os
import sys

# Add the parent directory to path to import hcclang
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

def test_simple_allreduce_program():
    """Test creating a simple AllReduce program with actual operations."""
    print("=== Testing Simple AllReduce Program ===")
    
    try:
        from hcclang.language import HCCLProgram, chunk
        from hcclang.language.collectives import AllReduce
        from hcclang.language.ir import Buffer
        from hcclang.topologies import generic
        
        # Create a simple ring topology with 4 nodes
        topo = generic.ring(4)
        collective = AllReduce(4, 2, False)  # 4 ranks, 2 chunks, not inplace
        
        with HCCLProgram("simple_allreduce", topo, collective, 1) as prog:
            print(f"✓ Created AllReduce program: {prog.name}")
            
            # Get input chunks for rank 0
            chunk0 = chunk(0, Buffer.input, 0, 1)
            chunk1 = chunk(0, Buffer.input, 1, 1)
            
            print(f"✓ Got input chunks: {chunk0}, {chunk1}")
            
            # Simple ring allreduce: send to next rank
            next_rank = (0 + 1) % prog.num_ranks
            copy_result = chunk0.copy(next_rank, Buffer.input, 0)
            print(f"✓ Copied chunk from rank 0 to rank {next_rank}: {copy_result}")
            
            # Test that the copy was recorded in the instruction DAG
            operations_count = len(prog.instr_dag.operations)
            print(f"✓ Operations recorded in DAG: {operations_count}")
            
        return True
        
    except Exception as e:
        print(f"❌ Simple AllReduce program test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_copy_between_ranks():
    """Test copy operations between different ranks."""
    print("\n=== Testing Copy Operations Between Ranks ===")
    
    try:
        from hcclang.language import HCCLProgram, chunk
        from hcclang.language.collectives import AllReduce
        from hcclang.language.ir import Buffer
        from hcclang.topologies import generic
        
        # Use fully connected topology for easier copying
        topo = generic.fully_connected(3)
        collective = AllReduce(3, 4, False)
        
        with HCCLProgram("copy_test", topo, collective, 1) as prog:
            # Get chunks from different ranks
            src_chunk = chunk(0, Buffer.input, 0, 1)
            
            # Copy to rank 1
            dst_chunk1 = src_chunk.copy(1, Buffer.input, 0)
            print(f"✓ Copied from rank 0 to rank 1: {dst_chunk1}")
            
            # Copy to rank 2
            dst_chunk2 = src_chunk.copy(2, Buffer.input, 0)
            print(f"✓ Copied from rank 0 to rank 2: {dst_chunk2}")
            
            # Verify buffer contents were updated
            original_data = prog.buffers[0][Buffer.input][0]
            copied_data1 = prog.buffers[1][Buffer.input][0]
            copied_data2 = prog.buffers[2][Buffer.input][0]
            
            print(f"✓ Original data: {original_data}")
            print(f"✓ Copied data (rank 1): {copied_data1}")
            print(f"✓ Copied data (rank 2): {copied_data2}")
            
            # All should be the same
            assert original_data == copied_data1 == copied_data2
            print("✓ Data consistency verified")
            
        return True
        
    except Exception as e:
        print(f"❌ Copy operations test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_reduce_operations():
    """Test reduce operations between chunks."""
    print("\n=== Testing Reduce Operations ===")
    
    try:
        from hcclang.language import HCCLProgram, chunk
        from hcclang.language.collectives import AllReduce
        from hcclang.language.ir import Buffer
        from hcclang.topologies import generic
        
        topo = generic.fully_connected(2)
        collective = AllReduce(2, 2, False)
        
        with HCCLProgram("reduce_test", topo, collective, 1) as prog:
            # Get chunks from both ranks
            chunk_rank0 = chunk(0, Buffer.input, 0, 1)
            chunk_rank1 = chunk(1, Buffer.input, 0, 1)
            
            print(f"✓ Source chunks: {chunk_rank0}, {chunk_rank1}")
            
            # Copy rank1's chunk to rank0, then reduce
            copied_chunk = chunk_rank1.copy(0, Buffer.input, 1)
            target_chunk = chunk(0, Buffer.input, 1, 1)
            
            # Perform reduce operation
            reduced_chunk = target_chunk.reduce(chunk_rank0)
            print(f"✓ Reduce operation completed: {reduced_chunk}")
            
            # Check that operations were recorded
            operations_count = len(prog.instr_dag.operations)
            print(f"✓ Total operations in DAG: {operations_count}")
            
        return True
        
    except Exception as e:
        print(f"❌ Reduce operations test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_xml_generation():
    """Test XML generation from a simple program."""
    print("\n=== Testing XML Generation ===")
    
    try:
        from hcclang.language import HCCLProgram, chunk, XML
        from hcclang.language.collectives import AllReduce
        from hcclang.language.ir import Buffer
        from hcclang.topologies import generic
        
        topo = generic.ring(2)  # Simple 2-node ring
        collective = AllReduce(2, 1, False)  # 2 ranks, 1 chunk
        
        with HCCLProgram("xml_test", topo, collective, 1) as prog:
            # Create a simple operation
            src_chunk = chunk(0, Buffer.input, 0, 1)
            dst_chunk = src_chunk.copy(1, Buffer.input, 0)
            
            print("✓ Created simple program with one copy operation")
            
            # Generate XML
            xml_output = prog.generate_xml()
            print("✓ XML generation completed")
            print(f"✓ XML output length: {len(xml_output)} characters")
            
            # Check that XML contains expected elements
            assert "<algo" in xml_output, "XML should contain algo element"
            assert "gpu" in xml_output, "XML should contain gpu elements"
            print("✓ XML structure validation passed")
            
        return True
        
    except Exception as e:
        print(f"❌ XML generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_program_lowering():
    """Test program lowering to IR."""
    print("\n=== Testing Program Lowering ===")
    
    try:
        from hcclang.language import HCCLProgram, chunk
        from hcclang.language.collectives import AllReduce
        from hcclang.language.ir import Buffer
        from hcclang.topologies import generic
        
        topo = generic.fully_connected(2)
        collective = AllReduce(2, 1, False)
        
        with HCCLProgram("lowering_test", topo, collective, 1) as prog:
            # Create some operations
            src_chunk = chunk(0, Buffer.input, 0, 1)
            dst_chunk = src_chunk.copy(1, Buffer.input, 0)
            
            print("✓ Created program with operations")
            
            # Test lowering
            ir_program = prog.lower()
            print("✓ Program lowering completed")
            print(f"✓ IR Program name: {ir_program.name}")
            print(f"✓ IR Program collective: {ir_program.collective}")
            print(f"✓ IR Program GPUs: {len(ir_program.gpus)}")
            
            # Check GPU structure
            for i, gpu in enumerate(ir_program.gpus):
                print(f"  - GPU {i}: rank={gpu.rank}, threadblocks={len(gpu.threadblocks)}")
                
        return True
        
    except Exception as e:
        print(f"❌ Program lowering test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_buffer_management_advanced():
    """Test advanced buffer management features."""
    print("\n=== Testing Advanced Buffer Management ===")
    
    try:
        from hcclang.language import HCCLProgram, chunk
        from hcclang.language.collectives import AllGather
        from hcclang.language.ir import Buffer
        from hcclang.topologies import generic
        
        topo = generic.fully_connected(3)
        collective = AllGather(3, 2, False)  # AllGather: 3 ranks, 2 chunks per rank
        
        with HCCLProgram("buffer_advanced_test", topo, collective, 1) as prog:
            print("✓ Created AllGather program")
            
            # Test scratch buffer creation
            prog.check_buffer_exists(0, "temp_buffer")
            prog.check_buffer_exists(1, "temp_buffer")
            
            print("✓ Scratch buffers created")
            
            # Test buffer access patterns
            for rank in range(prog.num_ranks):
                input_buffer = prog.buffers[rank][Buffer.input]
                output_buffer = prog.buffers[rank][Buffer.output]
                
                print(f"  - Rank {rank}: input_size={len(input_buffer)}, output_size={len(output_buffer)}")
                
                # Test get_chunks method
                chunks = prog.get_chunks(rank, Buffer.input, 0, len(input_buffer))
                non_null_chunks = [c for c in chunks if c is not None]
                print(f"    Input chunks: {len(non_null_chunks)}/{len(chunks)} non-null")
            
        return True
        
    except Exception as e:
        print(f"❌ Advanced buffer management test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all advanced functionality tests."""
    print("HCCLang Language Module - Advanced Functionality Tests")
    print("=" * 70)
    
    test_functions = [
        test_simple_allreduce_program,
        test_copy_between_ranks,
        test_reduce_operations,
        test_xml_generation,
        test_program_lowering,
        test_buffer_management_advanced,
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
    print(f"Advanced Tests Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All advanced tests passed!")
        return True
    else:
        print("⚠️  Some tests failed. Please check the output above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 