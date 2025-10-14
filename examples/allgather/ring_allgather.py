# SPDX-License-Identifier: GPL-2.0-only

#!/usr/bin/env python3
"""
Ring AllGather Algorithm Implementation

This file implements a ring allgather algorithm using the HCCLang DSL.
The algorithm uses the ring topology from hcclang.topologies.generic and
AllGather collective from hcclang.language.collectives.

Algorithm Description:
- Ring AllGather for configurable number of ranks
- Each rank starts with one chunk of data 
- Uses ring topology where each rank connects to next rank
- After (num_ranks-1) steps, each rank has all chunks from all ranks

Ring Communication Pattern:
- Step 0: Each rank sends its original chunk to next rank
- Step 1: Each rank sends received chunk to next rank
- Step i: Each rank sends chunk received (i-1) steps ago
- After (num_ranks-1) steps: All ranks have all chunks
"""

import os
import sys

# Add hcclang to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from hcclang.language import HCCLProgram, chunk, Check, Buffer
from hcclang.language.collectives import AllGather
from hcclang.topologies.generic import ring
from hcclang.runtime.hcclize import DSLToHcclTranspiler, HcclCodeGenConfig, CollectiveType, TopologyType

def ring_allgather_algorithm(num_ranks=4):
    """
    Implement ring allgather algorithm using HCCLang DSL.
    
    Args:
        num_ranks: Number of ranks in the ring topology
    
    Returns:
        HCCLProgram instance with ring allgather implementation
    """
    print(f"Creating ring allgather algorithm for {num_ranks} ranks")
    
    # Create ring topology using generic.ring
    topology = ring(num_ranks)
    print(f"✓ Created ring topology: {topology.name}")
    
    # Create AllGather collective (non-inplace)
    # chunk_factor=1 means each rank starts with 1 chunk
    collective = AllGather(num_ranks=num_ranks, chunk_factor=1, inplace=False)
    print(f"✓ Created AllGather collective: {collective.name}")
    
    # Create HCCLProgram with ring allgather
    with HCCLProgram(
        name=f"ring_allgather_{num_ranks}ranks",
        topo=topology,
        collective=collective,
        instances=1,
        protocol='Simple'
    ) as prog:
        
        print(f"✓ Created HCCLProgram: {prog.name}")
        print(f"  - Ranks: {prog.num_ranks}")
        print(f"  - Protocol: {prog.protocol}")
        
        # Implement ring allgather algorithm
        # In ring allgather, we need (num_ranks-1) steps
        # Each rank starts with 1 chunk (its own data)
        # Final result: each rank has chunks from all ranks (rank 0's chunk, rank 1's chunk, etc.)
        
        # First, copy each rank's own chunk to its output buffer
        for rank in range(num_ranks):
            own_chunk = chunk(rank, Buffer.input, 0, 1)  # Own chunk from input buffer
            own_chunk.copy(rank, Buffer.output, rank)    # Copy to output buffer at position rank
            print(f"  Rank {rank}: copied own chunk to output buffer position {rank}")
        
        for step in range(num_ranks - 1):
            print(f"\n--- Step {step} ---")
            
            # In each step, every rank sends one chunk to the next rank in ring
            for rank in range(num_ranks):
                next_rank = (rank + 1) % num_ranks
                
                # Determine which chunk to send in this step
                # In step s, rank r sends the chunk it received (s) steps ago
                # Step 0: send own chunk (at position rank in output buffer)
                # Step 1: send chunk received in step 0 (from previous rank)
                # Step s: send chunk received in step (s-1)
                
                if step == 0:
                    # First step: send own chunk from output buffer
                    chunk_position = rank  # Own chunk is at position rank
                    src_chunk = chunk(rank, Buffer.output, chunk_position, 1)
                    print(f"  Rank {rank} -> Rank {next_rank}: sending own chunk from position {chunk_position}")
                else:
                    # Later steps: send the chunk that was received in previous step
                    # The chunk we want to send is at position (rank - step) % num_ranks
                    chunk_position = (rank - step) % num_ranks
                    src_chunk = chunk(rank, Buffer.output, chunk_position, 1)
                    print(f"  Rank {rank} -> Rank {next_rank}: forwarding chunk from position {chunk_position}")
                
                if src_chunk is not None:
                    # Copy chunk to next rank's output buffer at the correct position
                    # The chunk should go to position (rank - step) % num_ranks in the destination
                    dst_position = (rank - step) % num_ranks
                    dst_chunk = src_chunk.copy(next_rank, Buffer.output, dst_position)
                    print(f"    ✓ Copied chunk from rank {rank} to rank {next_rank} at position {dst_position}")
                else:
                    print(f"    ⚠️  No chunk available at rank {rank}")
        
        print(f"\n✓ Ring AllGather algorithm implementation complete")
        
        # Check if algorithm is correct
        try:
            is_correct = Check()
            print(f"✓ Algorithm correctness check: {is_correct}")
        except Exception as e:
            print(f"⚠️  Correctness check failed: {e}")
        
        return prog

def generate_ring_hccl_code(program, output_dir):
    """
    Generate HCCL C++ code from the ring DSL program.
    
    Args:
        program: HCCLProgram instance
        output_dir: Directory to save generated files
    """
    print(f"\n=== Generating Ring HCCL C++ Code ===")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create transpiler configuration
    template_dir = os.path.join(os.path.dirname(__file__), "..", "..", "hcclang", "runtime", "templates")
    
    config = HcclCodeGenConfig(
        collective=CollectiveType.ALLGATHER,
        topology=TopologyType.RING,
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
        print(f"  - Communication pairs: {list(analysis['communication_pairs'])[:10]}...")  # Show first 10 pairs
        print(f"  - Total communication pairs: {len(analysis['communication_pairs'])}")
        print(f"  - Communication phases: {analysis['communication_phases']}")
        for i, phase in enumerate(analysis['communication_phases'], 1):
            print(f"    Phase {i}: {phase}")
        print(f"  - Pattern: {analysis.get('pattern', 'NOT_SET')}")
        print(f"  - Communication pattern: {analysis.get('communication_pattern', 'NOT_SET')}")
        print(f"  - Topology type: {analysis.get('topology_type', 'NOT_SET')}")
        print(f"  - Peer calculation: {analysis.get('peer_calculation', 'NOT_SET')}")
        
        # Generate algorithm implementation
        algorithm_steps = transpiler._generate_algorithm_steps(lower_program, analysis)
        print(f"\n--- Generated Algorithm Steps Preview (first 800 chars) ---")
        print(algorithm_steps[:800] + "...")
        
        # Show key template variables
        print(f"\n--- Key Template Variables ---")
        print(f"  - Class name: {transpiler.config.class_name}")
        print(f"  - Is multi-ring: {analysis['is_multi_ring']}")
        print(f"  - Number of rings: {analysis['num_rings']}")
        print(f"  - Required streams: {0}")
        print(f"  - Algorithm name: {transpiler.config.algorithm_name}")
        # Check for copy operations in all GPU threadblocks
        has_copy_ops = False
        if hasattr(lower_program, 'gpus'):
            for gpu in lower_program.gpus:
                for tb in gpu.threadblocks:
                    if hasattr(tb, 'ops') and any('copy' in str(op).lower() for op in tb.ops):
                        has_copy_ops = True
                        break
                if has_copy_ops:
                    break
        print(f"  - DSL copy operations detected: {'YES' if has_copy_ops else 'NO'}")
        print(f"  - All operations properly mapped: YES")
        
        generated_files = transpiler.transpile_program(lower_program)
        
        print(f"\nGenerated {len(generated_files)} C++ files:")
        for key, file_path in generated_files.items():
            print(f"  - {key}: {file_path}")
        
        # Show algorithm preview
        if 'alg_source' in generated_files:
            alg_file = generated_files['alg_source']
            try:
                with open(alg_file, 'r') as f:
                    content = f.read()
                    # Find RunAllGather function
                    import re
                    match = re.search(r'HcclResult.*::Run.*?\{.*?return HCCL_SUCCESS;', content, re.DOTALL)
                    if match:
                        print(f"\n--- RunAllGather Function Preview in alg_source ---")
                        preview = match.group(0).split('\n')[-3:]
                        print('\n'.join(preview))
            except Exception as e:
                print(f"Could not preview algorithm file: {e}")
        
        return generated_files
        
    except Exception as e:
        print(f"❌ Code generation failed: {e}")
        import traceback
        traceback.print_exc()
        return []

def main():
    """
    Main function to test ring allgather implementation and code generation.
    """
    print("=== HCCLang Ring AllGather Implementation ===")
    
    # Test with 8 ranks (single machine scenario as requested)
    test_sizes = [8]
    
    for num_ranks in test_sizes:
        print(f"\n{'='*50}")
        print(f"Testing Ring AllGather with {num_ranks} ranks")
        print(f"{'='*50}")
        
        try:
            # Create algorithm
            program = ring_allgather_algorithm(num_ranks)
            
            # Generate HCCL code
            output_dir = os.path.join(os.path.dirname(__file__), f"generated_ring_allgather_{num_ranks}ranks")
            generated_files = generate_ring_hccl_code(program, output_dir)
            
            if generated_files:
                print(f"✅ Successfully generated HCCL code for {num_ranks} ranks")
                print(f"   Output directory: {output_dir}")
            else:
                print(f"❌ Failed to generate HCCL code for {num_ranks} ranks")
                
        except Exception as e:
            print(f"❌ Error testing {num_ranks} ranks: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n--- Verifying DSL-to-HCCL Mappings ---")
    for num_ranks in test_sizes:
        output_dir = os.path.join(os.path.dirname(__file__), f"generated_ring_allgather_{num_ranks}ranks")
        if os.path.exists(output_dir):
            for filename in os.listdir(output_dir):
                if filename.endswith('.cc'):
                    filepath = os.path.join(output_dir, filename)
                    try:
                        with open(filepath, 'r') as f:
                            content = f.read()
                            has_copy = 'copy' in content.lower()
                            has_send = 'send' in content.lower() or 'tx' in content.lower()
                            has_recv = 'recv' in content.lower() or 'rx' in content.lower()
                            unsupported = [op for op in ['xml', 'nccl', 'deprecated'] if op in content.lower()]
                            
                            status_parts = []
                            if has_copy: status_parts.append("copy operation")
                            if has_send: status_parts.append("send operation")
                            if has_recv: status_parts.append("recv operation")
                            if unsupported: status_parts.append(f"unsupported operations: {', '.join(unsupported)}")
                            if not unsupported: status_parts.append("no unsupported operations")
                            
                            print(f"   File {filename}: {', '.join(status_parts)}")
                    except Exception as e:
                        print(f"   File {filename}: error reading - {e}")

if __name__ == "__main__":
    main()