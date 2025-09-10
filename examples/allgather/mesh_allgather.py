#!/usr/bin/env python3
"""
Mesh AllGather Algorithm Implementation

This file implements a mesh allgather algorithm using the HCCLang DSL.
The algorithm uses the fully_connected topology from hcclang.topologies.generic
as a mesh, where each rank can communicate with all other ranks directly.

Algorithm Description:
- Mesh AllGather for configurable number of ranks using fully connected topology
- Each rank starts with one chunk of data
- Uses mesh/fully_connected topology where each rank connects to all other ranks
- Efficient parallel data exchange pattern suitable for small to medium clusters

Mesh Communication Pattern:
- All ranks can communicate simultaneously in parallel
- Each rank broadcasts its data to all other ranks
- Highly parallel but may have higher network contention
- Optimal for scenarios with high bandwidth, low latency networks
"""

import os
import sys

# Add hcclang to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from hcclang.language import HCCLProgram, chunk, Check, Buffer
from hcclang.language.collectives import AllGather
from hcclang.topologies.generic import fully_connected
from hcclang.runtime.hcclize import DSLToHcclTranspiler, HcclCodeGenConfig, CollectiveType, TopologyType

def mesh_allgather_algorithm(num_ranks=4):
    """
    Implement mesh allgather algorithm using HCCLang DSL.
    
    Args:
        num_ranks: Number of ranks in the mesh topology
    
    Returns:
        HCCLProgram instance with mesh allgather implementation
    """
    print(f"Creating mesh allgather algorithm for {num_ranks} ranks")
    
    # Create fully connected topology as mesh
    topology = fully_connected(num_ranks)
    print(f"✓ Created mesh topology: {topology.name}")
    
    # Create AllGather collective (non-inplace)
    # chunk_factor=1 means each rank starts with 1 chunk
    collective = AllGather(num_ranks=num_ranks, chunk_factor=1, inplace=False)
    print(f"✓ Created AllGather collective: {collective.name}")
    
    # Create HCCLProgram with mesh allgather implementation
    with HCCLProgram(
        name=f"mesh_allgather_{num_ranks}ranks",
        topo=topology,
        collective=collective,
        instances=1,
        protocol='Simple'
    ) as prog:
        print(f"✓ Created HCCLProgram: {prog.name}")
        print(f"  - Ranks: {prog.num_ranks}")
        print(f"  - Protocol: {prog.protocol}")
        
        # Implement mesh allgather algorithm
        # In mesh (fully connected) allgather, all ranks can communicate simultaneously
        # Each rank receives data from all other ranks in parallel
        
        # Step 1: Each rank copies its own data to output buffer
        for rank in range(num_ranks):
            own_chunk = chunk(rank, Buffer.input, 0, 1)  # Own chunk from input buffer
            own_chunk.copy(rank, Buffer.output, rank)    # Copy to output buffer at position rank
            print(f"  Rank {rank}: copied own chunk to output buffer position {rank}")
        
        # Step 2: All-to-all data exchange in mesh topology
        # In mesh topology, each rank can communicate with all other ranks simultaneously
        # We'll implement a simplified mesh pattern where each rank receives data from all others
        for step in range(num_ranks - 1):
            print(f"\n--- Step {step} ---")
            for rank in range(num_ranks):
                # Each rank receives from one other rank per step in round-robin fashion
                src_rank = (rank + step + 1) % num_ranks
                
                # In mesh allgather, we need to simulate receiving data from src_rank
                # Create a receive operation from src_rank to current rank
                # The chunk being sent is from src_rank's original position
                src_chunk = chunk(src_rank, Buffer.output, src_rank, 1)  # Source data from src_rank
                
                # Copy the chunk to current rank's output buffer at the source's position
                dst_chunk = src_chunk.copy(rank, Buffer.output, src_rank)
                print(f"  Rank {rank} <- Rank {src_rank}: mesh communication")
                print(f"    ✓ Received chunk from rank {src_rank} at position {src_rank}")
        
        print(f"\n✓ Mesh AllGather algorithm implementation complete")
        
        return prog

def main():
    """Test mesh allgather with 8 ranks."""
    print("=== HCCLang Mesh AllGather Implementation ===")
    print()
    
    num_ranks = 8
    print("=" * 50)
    print(f"Testing Mesh AllGather with {num_ranks} ranks")
    print("=" * 50)
    
    # Create the algorithm
    program = mesh_allgather_algorithm(num_ranks)
    
    print()
    print("=== Generating Mesh HCCL C++ Code ===")
    
    # Configure code generation
    output_dir = f"generated_mesh_allgather_{num_ranks}ranks"
    template_dir = os.path.join(os.path.dirname(__file__), "..", "..", "hcclang", "runtime", "templates")
    
    config = HcclCodeGenConfig(
        collective=CollectiveType.ALLGATHER,
        topology=TopologyType.MESH,
        output_dir=output_dir,
        template_dir=template_dir,
        algorithm_name=program.name,
        num_ranks=program.num_ranks,
        num_steps=0  # Will be calculated from program
    )
    
    # Initialize transpiler
    transpiler = DSLToHcclTranspiler(config)
    
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate C++ code files
        # First convert HCCLProgram to lower-level Program representation
        lower_program = program.lower()
        
        # Debug: Print analysis results from enhanced transpiler
        print(f"\n--- Transpiler Analysis Debug ---")
        analysis = transpiler._analyze_communication_pattern(lower_program)
        print(f"DSL Program Analysis Results:")
        print(f"  - Total steps: {analysis['total_steps']}")
        print(f"  - Max rank: {analysis['max_rank']}")
        print(f"  - Number of mesh connections: {len(analysis['communication_pairs'])}")
        print(f"  - Communication pairs: {list(analysis['communication_pairs'])[:10]}...")  # Show first 10 pairs
        print(f"  - Total communication pairs: {len(analysis['communication_pairs'])}")
        print(f"  - Communication phases: {analysis['communication_phases']}")
        for i, phase in enumerate(analysis['communication_phases'], 1):
            print(f"    Phase {i}: {phase}")
        print(f"  - Pattern: {analysis.get('pattern', 'NOT_SET')}")
        print(f"  - Communication pattern: {analysis.get('communication_pattern', 'NOT_SET')}")
        print(f"  - Topology type: {analysis.get('topology_type', 'NOT_SET')}")
        print(f"  - Peer calculation: {analysis.get('peer_calculation', 'NOT_SET')}")
        
        # Generate code using the transpiler
        generated_files = transpiler.transpile_program(lower_program)
        
        print(f"\n--- Generated Algorithm Steps Preview (first 800 chars) ---")
        if 'alg_source' in generated_files:
            try:
                with open(generated_files['alg_source'], 'r') as f:
                    content = f.read()
                    # Find the algorithm implementation
                    if "AllGather Algorithm Implementation" in content:
                        start = content.find("AllGather Algorithm Implementation")
                        preview = content[start:start+800]
                        print(preview)
                    else:
                        print(content[:800])
            except Exception as e:
                print(f"Could not read generated file: {e}")
        
        print(f"\nGenerated {len(generated_files)} C++ files:")
        for file_type, file_path in generated_files.items():
            print(f"  - {file_type}: {file_path}")
        
        print(f"\n--- RunAllGather Function Preview in alg_source ---")
        if 'alg_source' in generated_files:
            try:
                with open(generated_files['alg_source'], 'r') as f:
                    content = f.read()
                    # Find RunAllGather function
                    if "RunAllGather" in content:
                        start = content.find("HcclResult AllgatherMesh::RunAllGather")
                        if start == -1:
                            start = content.find("RunAllGather")
                        end = content.find("}", start)
                        if end != -1:
                            preview = content[start:end+1]
                            # Show last 200 chars
                            print(preview[-200:])
                        else:
                            print("Could not find end of RunAllGather function")
                    else:
                        print("RunAllGather function not found")
            except Exception as e:
                print(f"Could not read generated file: {e}")
        
        print(f"✅ Successfully generated HCCL code for {num_ranks} ranks")
        print(f"   Output directory: {os.path.dirname(generated_files.get('alg_source', ''))}")
        
        print(f"\n--- Verifying DSL-to-HCCL Mappings ---")
        for file_type, file_path in generated_files.items():
            if file_path.endswith('.cc'):
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                        operations = []
                        if "copy" in content.lower():
                            operations.append("copy operation")
                        if "send" in content.lower():
                            operations.append("send operation")
                        if "recv" in content.lower():
                            operations.append("recv operation")
                        if "txasync" in content.lower():
                            operations.append("txasync operation")
                        if "rxasync" in content.lower():
                            operations.append("rxasync operation")
                        
                        unsupported = []
                        if "TODO" in content:
                            unsupported.append("TODO markers")
                        if "NOT_IMPLEMENTED" in content:
                            unsupported.append("NOT_IMPLEMENTED")
                        
                        operations_str = ", ".join(operations) if operations else "no operations"
                        unsupported_str = ", ".join(unsupported) if unsupported else "no unsupported operations"
                        print(f"   File {os.path.basename(file_path)}: {operations_str}, {unsupported_str}")
                except Exception as e:
                    print(f"   File {os.path.basename(file_path)}: could not analyze - {e}")
        
    except Exception as e:
        print(f"❌ Code generation failed: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()