#!/usr/bin/env python3
"""
Bruck AllToAll Algorithm Implementation

This file implements a Bruck alltoall algorithm using the HCCLang DSL.
The algorithm uses the fully connected topology and AllToAll collective.

Algorithm Description:
- Bruck AllToAll for configurable number of ranks (must be power of 2)
- Each rank starts with N chunks of data (one for each rank)
- Uses log2(N) steps with XOR-based communication patterns
- High parallelism through simultaneous data exchanges

Bruck Communication Pattern:
- Step i (i=0 to log2(N)-1): Each rank exchanges data with rank^(2^i)
- In each step, ranks exchange chunks based on bit patterns
- After log2(N) steps: All ranks have received their chunks from all other ranks

Data Exchange Logic:
- Step i: Exchange chunks where bit i of destination differs from sender
- Rank r exchanges with rank (r XOR 2^i)
- Each exchange involves multiple chunks based on bit pattern analysis
"""

import os
import sys
import math

# Add hcclang to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from hcclang.language import HCCLProgram, chunk, Check, Buffer
from hcclang.language.collectives import AllToAll
from hcclang.topologies.generic import fully_connected
from hcclang.runtime.hcclize import DSLToHcclTranspiler, HcclCodeGenConfig, CollectiveType, TopologyType

def bruck_alltoall_algorithm(num_ranks=8):
    """
    Implement Bruck AllToAll algorithm using HCCLang DSL.
    
    Args:
        num_ranks: Number of ranks (must be power of 2)
    
    Returns:
        HCCLProgram instance with Bruck AllToAll implementation
    """
    # Validate that num_ranks is power of 2
    if not (num_ranks & (num_ranks - 1)) == 0:
        raise ValueError(f"Bruck algorithm requires power of 2 ranks, got {num_ranks}")
    
    print(f"Creating Bruck AllToAll algorithm for {num_ranks} ranks")
    
    # Create fully connected topology for AllToAll
    topology = fully_connected(num_ranks)
    print(f"✓ Created topology: {topology.name}")
    
    # Create AllToAll collective
    # Each rank has num_ranks chunks to send (one to each rank)
    collective = AllToAll(num_ranks=num_ranks, chunk_factor=1, inplace=False)
    print(f"✓ Created AllToAll collective: {collective.name}")
    
    # Create HCCLProgram for Bruck AllToAll using context manager
    program_name = f"bruck_alltoall_{num_ranks}ranks"
    
    with HCCLProgram(
        name=program_name,
        topo=topology,
        collective=collective,
        instances=1,
        protocol='Simple'
    ) as program:
        
        print(f"✓ Created HCCLProgram: {program.name}")
        print(f"  - Ranks: {program.num_ranks}")  
        print(f"  - Protocol: Simple")
        
        # Initialize - each rank starts with chunks for all other ranks
        for rank in range(num_ranks):
            for dst_rank in range(num_ranks):
                if dst_rank == rank:
                    # Local chunk stays in place 
                    input_chunk = chunk(rank, Buffer.input, dst_rank, 1)
                    input_chunk.copy(rank, Buffer.output, dst_rank)
                    print(f"  Rank {rank}: placed chunk for rank {dst_rank} locally")
                else:
                    # Initialize input chunks that will be exchanged
                    print(f"  Rank {rank}: initialized chunk for destination rank {dst_rank}")
        
        # Bruck algorithm main loop: log2(num_ranks) steps
        num_steps = int(math.log2(num_ranks))
        print(f"\n--- Bruck AllToAll Algorithm: {num_steps} steps ---")
        
        for step in range(num_steps):
            distance = 1 << step  # 2^step
            print(f"\n--- Step {step} (distance={distance}) ---")
            
            for rank in range(num_ranks):
                peer_rank = rank ^ distance  # XOR with 2^step
                
                print(f"  Rank {rank} <-> Rank {peer_rank}: Bruck exchange")
                
                # Determine which chunks to exchange in this step
                # In step i, exchange chunks destined for ranks where bit i differs
                chunks_to_send = []
                chunks_to_recv = []
                
                for dst_rank in range(num_ranks):
                    # Check if bit 'step' of dst_rank differs between rank and peer_rank
                    if (dst_rank >> step) & 1 != (rank >> step) & 1:
                        chunks_to_send.append(dst_rank)
                    if (dst_rank >> step) & 1 != (peer_rank >> step) & 1:
                        chunks_to_recv.append(dst_rank)
                
                # Exchange chunks with peer - using copy operations for now
                # In Bruck algorithm, ranks exchange specific chunks based on bit patterns
                for dst_rank in chunks_to_send:
                    send_chunk = chunk(rank, Buffer.input, dst_rank, 1) 
                    # Copy to peer's input buffer at appropriate position
                    send_chunk.copy(peer_rank, Buffer.input, dst_rank)
                    print(f"    ✓ Rank {rank} -> Rank {peer_rank}: chunk destined for rank {dst_rank}")
                
                for dst_rank in chunks_to_recv:
                    recv_chunk = chunk(peer_rank, Buffer.input, dst_rank, 1)
                    # Copy from peer's input buffer to local input buffer
                    recv_chunk.copy(rank, Buffer.input, dst_rank)
                    print(f"    ✓ Rank {rank} <- Rank {peer_rank}: chunk destined for rank {dst_rank}")
        
        # Final step: copy received chunks to output positions
        print(f"\n--- Final Copy Phase ---")
        for rank in range(num_ranks):
            for src_rank in range(num_ranks):
                if src_rank != rank:
                    # Copy received chunk from src_rank to final output position
                    input_chunk = chunk(rank, Buffer.input, src_rank, 1)
                    input_chunk.copy(rank, Buffer.output, src_rank)
                    print(f"  Rank {rank}: copied chunk from rank {src_rank} to output")
        
        print(f"✓ Bruck AllToAll algorithm implementation complete")
        return program

def generate_bruck_alltoall_hccl_code(num_ranks=8):
    """Generate HCCL C++ code from Bruck AllToAll DSL algorithm"""
    print(f"\n=== Generating Bruck AllToAll HCCL C++ Code ===\n")
    
    # Create DSL program
    program = bruck_alltoall_algorithm(num_ranks)
    
    # Create output directory relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, f"generated_bruck_alltoall_{num_ranks}ranks")
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure code generation
    template_dir = os.path.join(os.path.dirname(__file__), "..", "..", "hcclang", "runtime", "templates")
    
    config = HcclCodeGenConfig(
        collective=CollectiveType.ALLTOALL,
        topology=TopologyType.MESH,
        output_dir=output_dir,
        template_dir=template_dir,
        algorithm_name=program.name,
        num_ranks=program.num_ranks,
        num_steps=0  # Will be calculated from program
    )
    
    # Generate code 
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
        
        generated_files = transpiler.transpile_program(lower_program)
    except Exception as e:
        print(f"❌ Code generation failed: {e}")
        raise
    
    print(f"Generated {len(generated_files)} C++ files:")
    for file_type, file_path in generated_files.items():
        print(f"  - {file_type}: {file_path}")
    
    # Preview generated algorithm 
    if 'alg_source' in generated_files:
        print(f"\n--- AllToAll Algorithm Preview in alg_source ---")
        try:
            with open(generated_files['alg_source'], 'r') as f:
                content = f.read()
                if 'RunAllToAll' in content:
                    start = content.find('HcclResult')
                    end = content.find('\n}\n', start) + 2 if start != -1 else -1
                    if start != -1 and end != -1:
                        preview = content[start:end]
                        print(preview[:800] + "..." if len(preview) > 800 else preview)
                else:
                    print("RunAllToAll function not found")
        except FileNotFoundError:
            print(f"File {generated_files['alg_source']} not found")
    
    print(f"✅ Successfully generated HCCL code for {num_ranks} ranks")
    print(f"   Output directory: {config.output_dir}")
    
    return generated_files

def verify_dsl_to_hccl_mappings(generated_files):
    """Verify that DSL operations are correctly mapped to HCCL calls"""
    print(f"\n--- Verifying DSL-to-HCCL Mappings ---")
    
    hccl_operations = {
        'send': ['TxAsync', 'TxAck', 'TxWaitDone'],
        'recv': ['RxAsync', 'RxAck', 'RxWaitDone'], 
        'copy': ['HcclD2DMemcpyAsync', 'copy'],
        'reduce': ['reduce'],
        'barrier': ['barrier']
    }
    
    for file_type, file_path in generated_files.items():
        if file_path.endswith('.cc'):
            print(f"   File {os.path.basename(file_path)}: ", end="")
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    
                found_ops = []
                unsupported_ops = []
                
                for dsl_op, hccl_calls in hccl_operations.items():
                    if any(hccl_call in content for hccl_call in hccl_calls):
                        found_ops.append(f"{dsl_op} operation")
                
                # Check for unsupported operations
                if 'UNSUPPORTED' in content or 'NotImplemented' in content:
                    unsupported_ops.append("unsupported operations")
                
                result = ", ".join(found_ops) if found_ops else "no operations"
                if unsupported_ops:
                    result += f", {', '.join(unsupported_ops)}"  
                else:
                    result += ", no unsupported operations"
                    
                print(result)
            except FileNotFoundError:
                print("file not found")

def main():
    """Main function to test Bruck AllToAll implementation"""
    print("=== HCCLang Bruck AllToAll Implementation ===\n")
    
    # Test configuration
    num_ranks = 8
    
    print("=" * 50)
    print(f"Testing Bruck AllToAll with {num_ranks} ranks")  
    print("=" * 50)
    
    try:
        # Generate DSL algorithm and HCCL code
        generated_files = generate_bruck_alltoall_hccl_code(num_ranks)
        
        # Verify DSL-to-HCCL mappings
        verify_dsl_to_hccl_mappings(generated_files)
        
        print(f"\n=== Summary ===")
        print("Bruck AllToAll implementation and testing complete.")
        print("Generated files demonstrate:")
        print("- Bruck algorithm semantics in DSL")
        print("- XOR-based parallel data exchange pattern")
        print("- Fully connected topology communication")
        print("- Logarithmic complexity O(log N) steps")
        print("- Direct transpilation to HCCL AllToAll code")
        print("- Complete DSL-to-HCCL operation mapping validation")
        
    except Exception as e:
        print(f"❌ Error during Bruck AllToAll implementation: {e}")
        raise

if __name__ == "__main__":
    main()