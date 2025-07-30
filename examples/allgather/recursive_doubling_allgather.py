#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recursive Doubling AllGather Algorithm Implementation

This file implements a recursive doubling allgather algorithm using the HCCLang DSL.
The algorithm uses a binary tree pattern where each rank exchanges data with peers
at exponentially increasing distances, doubling the amount of data each iteration.

Algorithm Description:
- Recursive doubling AllGather for power-of-2 number of ranks
- Each iteration doubles the data size and halves the communication distance
- Log(n) iterations to complete, optimal for small to medium cluster sizes
- Uses XOR pattern for peer selection: peer = rank ^ count

Recursive Doubling Communication Pattern:
- Iteration 0: Exchange 1 chunk with neighbor distance 1
- Iteration 1: Exchange 2 chunks with neighbor distance 2  
- Iteration 2: Exchange 4 chunks with neighbor distance 4
- ...until all data is distributed
"""

import os
import sys
import math

# Add hcclang to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from hcclang.language import HCCLProgram, chunk, Check, Buffer
from hcclang.language.collectives import AllGather
from hcclang.topologies.generic import ring, fully_connected
from hcclang.runtime.hcclize import DSLToHcclTranspiler, HcclCodeGenConfig, CollectiveType, TopologyType

def recursive_doubling_allgather_algorithm(num_ranks=8):
    """
    Implement recursive doubling allgather algorithm using HCCLang DSL.
    
    Args:
        num_ranks: Total number of ranks (must be power of 2)
    
    Returns:
        HCCLProgram instance with recursive doubling allgather implementation
    """
    # Verify num_ranks is power of 2
    if num_ranks & (num_ranks - 1) != 0:
        raise ValueError(f"num_ranks ({num_ranks}) must be a power of 2 for recursive doubling")
    
    log_ranks = int(math.log2(num_ranks))
    
    print(f"Creating recursive doubling allgather algorithm for {num_ranks} ranks")
    print(f"Algorithm will complete in {log_ranks} iterations")
    
    # Create fully connected topology (recursive doubling requires all-to-all communication)
    topology = fully_connected(num_ranks)
    print(f"Created topology: {topology.name}")
    
    # Create AllGather collective (non-inplace)
    # chunk_factor=1 means each rank starts with 1 chunk
    collective = AllGather(num_ranks=num_ranks, chunk_factor=1, inplace=False)
    print(f"Created AllGather collective: {collective.name}")
    
    # Create HCCLProgram with recursive doubling allgather
    with HCCLProgram(
        name=f"recursive_doubling_allgather_{num_ranks}ranks",
        topo=topology,
        collective=collective,
        instances=1,
        protocol='Simple'
    ) as prog:
        
        print(f"Created HCCLProgram: {prog.name}")
        print(f"  - Ranks: {prog.num_ranks}")
        print(f"  - Iterations: {log_ranks}")
        print(f"  - Protocol: {prog.protocol}")
        
        # Phase 1: Initialize - each rank copies own data to output buffer
        print(f"\n=== Phase 1: Initialize Own Data ===")
        for rank in range(num_ranks):
            own_chunk = chunk(rank, Buffer.input, 0, 1)  # Own chunk from input buffer
            own_chunk.copy(rank, Buffer.output, rank)    # Copy to output buffer at position rank
            print(f"  Rank {rank}: copied own chunk to output position {rank}")
        
        # Phase 2: Recursive doubling iterations
        print(f"\n=== Phase 2: Recursive Doubling Iterations ===")
        
        count = 1
        iteration = 0
        while count < num_ranks:
            print(f"\n--- Iteration {iteration}: Exchange {count} chunks with distance {count} peers ---")
            
            # Every rank exchanges count chunks with neighbor count away
            for rank in range(num_ranks):
                peer = rank ^ count  # XOR pattern for peer selection
                index = (rank // count) * count  # Starting index for chunk block
                
                print(f"  Rank {rank} exchanges with Rank {peer}: chunks {index} to {index + count - 1}")
                
                # Send own chunk block to peer using DSL copy operation
                for chunk_offset in range(count):
                    chunk_index = index + chunk_offset
                    if chunk_index < num_ranks:
                        # Create chunk reference for the data to send
                        src_chunk = chunk(rank, Buffer.output, chunk_index, 1)
                        
                        # Copy chunk to peer's output buffer
                        # Using sendtb and recvtb parameters to specify threadblocks
                        dst_chunk = src_chunk.copy(peer, Buffer.output, chunk_index, sendtb=peer, recvtb=rank)
                        
                        print(f"    Rank {rank} -> Rank {peer}: chunk {chunk_index}")
            
            count *= 2  # Double the chunk count for next iteration
            iteration += 1
        
        print(f"\nRecursive Doubling AllGather algorithm implementation complete")
        print(f"  - Total iterations: {iteration}")
        print(f"  - Final chunk count per rank: {count // 2}")
        
        # Check if algorithm is correct
        try:
            is_correct = Check()
            print(f"Algorithm correctness check: {is_correct}")
        except Exception as e:
            print(f"Correctness check failed: {e}")
        
        return prog

def generate_recursive_doubling_hccl_code(program, output_dir):
    """
    Generate HCCL C++ code from the recursive doubling DSL program.
    
    Args:
        program: HCCLProgram instance
        output_dir: Directory to save generated files
    """
    print(f"\n=== Generating Recursive Doubling HCCL C++ Code ===")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create transpiler configuration
    template_dir = os.path.join(os.path.dirname(__file__), "..", "..", "hcclang", "runtime", "templates")
    
    config = HcclCodeGenConfig(
        collective=CollectiveType.ALLGATHER,
        topology=TopologyType.MESH,  # Recursive doubling requires fully connected/mesh topology
        output_dir=output_dir,
        template_dir=template_dir,
        algorithm_name=program.name,
        num_ranks=program.num_ranks,
        num_steps=0  # Will be calculated from program
    )
    
    # Initialize HCCL transpiler
    transpiler = DSLToHcclTranspiler(config)
    
    try:
        # Generate C++ code files
        # First convert HCCLProgram to lower-level Program representation
        lower_program = program.lower()
        
        # Debug: Print analysis results from enhanced transpiler
        print(f"\n--- Transpiler Analysis Debug ---")
        analysis = transpiler._analyze_communication_pattern(lower_program)
        print(f"DSL Program Analysis Results:")
        print(f"  - Total steps: {analysis['total_steps']}")
        print(f"  - Max rank: {analysis['max_rank']}")
        print(f"  - Number of rings: {analysis['num_rings']}")
        print(f"  - Is multi-ring: {analysis['is_multi_ring']}")
        print(f"  - Is hierarchical: {analysis['is_hierarchical']}")
        print(f"  - Ranks per ring: {analysis['ranks_per_ring']}")
        print(f"  - Communication phases: {len(analysis['communication_phases'])}")
        for i, phase in enumerate(analysis['communication_phases']):
            print(f"    Phase {i+1}: {phase['description']} ({phase['steps']} steps)")
        
        # Debug communication pattern detection
        print(f"  - Pattern: {analysis.get('pattern', 'NOT_SET')}")
        print(f"  - Communication pattern: {analysis.get('communication_pattern', 'NOT_SET')}")
        print(f"  - Topology type: {analysis.get('topology_type', 'NOT_SET')}")
        print(f"  - Peer calculation: {analysis.get('peer_calculation', 'NOT_SET')}")
        
        # Generate algorithm steps and show preview
        algorithm_steps = transpiler._generate_algorithm_steps(lower_program, analysis)
        print(f"\n--- Generated Algorithm Steps Preview (first 800 chars) ---")
        print(algorithm_steps[:800] + "..." if len(algorithm_steps) > 800 else algorithm_steps)
        
        # Generate executor orchestration code
        executor_orchestration = transpiler._generate_executor_orchestration(analysis)
        
        # Prepare template variables and show key ones
        template_vars = transpiler._prepare_template_variables(analysis, algorithm_steps, executor_orchestration)
        print(f"\n--- Key Template Variables ---")
        print(f"  - Class name: {template_vars['class_name']}")
        print(f"  - Is multi-ring: {template_vars['is_multi_ring']}")
        print(f"  - Number of rings: {template_vars['num_rings']}")
        print(f"  - Required streams: {template_vars['required_streams']}")
        print(f"  - Algorithm name: {template_vars['algorithm_name']}")
        
        # Check for DSL operations in the generated code
        if 'copy' in algorithm_steps.lower() or 'hccld2dmemcpyasync' in algorithm_steps.lower():
            print(f"  - DSL copy operations detected: YES")
        else:
            print(f"  - DSL copy operations detected: NO")
        
        if 'todo' in algorithm_steps.lower():
            print(f"  - Unsupported operations found: YES")
        else:
            print(f"  - All operations properly mapped: YES")
        
        generated_files = transpiler.transpile_program(lower_program)
        
        print(f"\nGenerated {len(generated_files)} C++ files:")
        for file_type, file_path in generated_files.items():
            print(f"  - {file_type}: {file_path}")
            
            # Show algorithm implementation preview for core files
            if 'alg' in file_type and file_path.endswith('.cc'):
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                        # Find RunAllGather function
                        start_idx = content.find('RunAllgather(')
                        if start_idx == -1:
                            start_idx = content.find('RunAllGather(')
                        if start_idx != -1:
                            end_idx = content.find('return HCCL_SUCCESS;', start_idx)
                            if end_idx != -1:
                                end_idx += len('return HCCL_SUCCESS;')
                                alg_function = content[start_idx:end_idx]
                                print(f"\n--- RunAllGather Function Preview in {file_type} ---")
                                preview_length = 500
                                print(alg_function[:preview_length] + "..." if len(alg_function) > preview_length else alg_function)
                except Exception as e:
                    print(f"  Warning: Could not preview algorithm function: {e}")
        
        return list(generated_files.values())
        
    except Exception as e:
        print(f"Code generation failed: {e}")
        import traceback
        traceback.print_exc()
        return []

def main():
    """
    Main function to test recursive doubling allgather implementations.
    """
    print("=== HCCLang Recursive Doubling AllGather Implementation ===")
    
    # Test single configuration first
    print(f"\nTesting Recursive Doubling AllGather: 8 ranks")
    print(f"=" * 60)
    
    try:
        # Create algorithm
        program = recursive_doubling_allgather_algorithm(8)
        
        # Generate HCCL code
        output_dir = os.path.join(os.path.dirname(__file__), "generated_recursive_doubling_allgather_8ranks")
        generated_files = generate_recursive_doubling_hccl_code(program, output_dir)
        
        if generated_files:
            print(f"Successfully generated HCCL code for 8 ranks")
            print(f"   Output directory: {output_dir}")
            
            # Check generated files contain proper DSL mappings
            print(f"\n--- Verifying DSL-to-HCCL Mappings ---")
            for file_path in generated_files:
                if file_path.endswith('.cc'):
                    try:
                        with open(file_path, 'r') as f:
                            content = f.read()
                            
                        dsl_mappings_found = []
                        if 'HcclD2DMemcpyAsync' in content:
                            dsl_mappings_found.append('copy operation')
                        if 'TxAsync' in content:
                            dsl_mappings_found.append('send operation')
                        if 'RxAsync' in content:
                            dsl_mappings_found.append('recv operation')
                        if 'TODO' not in content:
                            dsl_mappings_found.append('no unsupported operations')
                        
                        print(f"   File {os.path.basename(file_path)}: {', '.join(dsl_mappings_found) if dsl_mappings_found else 'basic structure only'}")
                    except Exception as e:
                        print(f"   File {os.path.basename(file_path)}: could not verify ({e})")
        else:
            print(f"Failed to generate HCCL code for 8 ranks")
            
    except Exception as e:
        print(f"Error in main test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()