#!/usr/bin/env python3
"""
Single Ring AllGather Algorithm Tutorial

This file demonstrates how to write a single ring allgather algorithm using HCCLang DSL,
and complete the full workflow from DSL to HCCL C++ code generation.

Algorithm Description:
- Single ring allgather for 4 ranks
- Each rank starts with one chunk of data
- After completion, each rank has all chunks from all ranks

Ring topology: 0 -> 1 -> 2 -> 3 -> 0
"""

import sys
import os

# Add the hcclang directory to Python path
tutorial_dir = os.path.dirname(__file__)
hcclang_demo_dir = os.path.dirname(tutorial_dir)
sys.path.insert(0, hcclang_demo_dir)

from hcclang.core.algorithm import Algorithm, Step
from hcclang.core.collectives import Collective, Chunk
from hcclang.topologies.generic import ring
from hcclang.topologies.topo_tools import Topology
from hcclang.solver.instance import Instance
from hcclang.runtime.serialization import save_hccl_object, load_hccl_object
from hcclang.runtime.ncclize import ncclize
from hcclang.runtime.hcclize_allgather import HcclAllGatherTemplateGenerator

def create_single_ring_allgather(num_ranks=4):
    """
    Create a single ring allgather algorithm using HCCLang DSL
    
    AllGather Phase Flow (configurable ranks, single ring):
    For n ranks, requires n-1 steps where each rank sends data to next rank in ring
    """
    
    # Define topology: single ring with configurable ranks
    topology = ring(num_ranks)
    
    # Initialize chunks and their pre/post conditions
    chunks = []
    for chunk_id in range(num_ranks):
        # Initially, chunk i is only on rank i
        precondition = {chunk_id}  # chunk_id is initially on rank chunk_id
        # Finally, all chunks are on all ranks
        postcondition = set(range(num_ranks))  # All ranks have this chunk
        chunks.append(Chunk(precondition, postcondition, address=chunk_id))
    
    # Define collective operation
    # Each rank initially has its own chunk, final result has all chunks on all ranks
    collective = Collective(name='allgather', 
                           num_nodes=num_ranks, 
                           chunks=chunks,  # Pass the chunks list
                           runtime_name='AllGather')
    
    # Define input/output mappings for each rank
    input_map = {}
    output_map = {}
    
    for rank in range(num_ranks):
        # Each rank initially has only its own chunk
        input_map[rank] = {rank}  # rank r has chunk r initially
        # After allgather, each rank has all chunks
        output_map[rank] = set(range(num_ranks))  # All ranks have all chunks
    
    # Create algorithm steps for single ring allgather
    steps = []
    
    # AllGather: (num_ranks-1) steps for ring topology
    for step in range(num_ranks - 1):
        sends = []
        for rank in range(num_ranks):
            src_rank = rank
            dst_rank = (rank + 1) % num_ranks
            
            # Determine which chunk to send in this step
            # In step s, rank r sends the chunk it received (s) steps ago
            chunk_to_send = (rank - step) % num_ranks
            
            sends.append([chunk_to_send, src_rank, dst_rank])
        
        steps.append(Step(rounds=1, sends=sends))
    
    # Create instance with algorithm metadata
    instance = Instance(steps=len(steps), chunks=num_ranks)
    
    # Create algorithm instance
    algorithm = Algorithm(
        name=f'single_ring_allgather_{num_ranks}rank',
        collective=collective,
        topology=topology,
        instance=instance,
        steps=steps,
        input_map=input_map,
        output_map=output_map
    )
    
    return algorithm

def create_multi_ring_allgather(num_ranks=8, num_rings=2):
    """
    Create a multi-ring allgather algorithm using HCCLang DSL
    
    Multi-Ring AllGather splits ranks across multiple rings to improve parallelism
    Each ring operates independently, then data is shared between rings
    """
    from hcclang.topologies.multi_ring import multi_ring
    
    # Ensure ranks can be evenly divided across rings
    if num_ranks % num_rings != 0:
        raise ValueError(f"num_ranks ({num_ranks}) must be divisible by num_rings ({num_rings})")
    
    ranks_per_ring = num_ranks // num_rings
    
    # Create multi-ring topology
    topology = multi_ring(num_ranks, num_rings)
    
    # Initialize chunks - each rank starts with one chunk
    chunks = []
    for chunk_id in range(num_ranks):
        precondition = {chunk_id}
        postcondition = set(range(num_ranks))
        chunks.append(Chunk(precondition, postcondition, address=chunk_id))
    
    # Define collective operation
    collective = Collective(name='allgather_multirig', 
                           num_nodes=num_ranks, 
                           chunks=chunks,
                           runtime_name='AllGather')
    
    # Define input/output mappings
    input_map = {}
    output_map = {}
    
    for rank in range(num_ranks):
        input_map[rank] = {rank}
        output_map[rank] = set(range(num_ranks))
    
    # Create algorithm steps for multi-ring allgather
    steps = []
    
    # Phase 1: AllGather within each ring
    for step in range(ranks_per_ring - 1):
        sends = []
        for ring_id in range(num_rings):
            ring_start = ring_id * ranks_per_ring
            for i in range(ranks_per_ring):
                src_rank = ring_start + i
                dst_rank = ring_start + (i + 1) % ranks_per_ring
                chunk_to_send = (src_rank - step) % num_ranks
                sends.append([chunk_to_send, src_rank, dst_rank])
        
        steps.append(Step(rounds=1, sends=sends))
    
    # Phase 2: Exchange data between rings (inter-ring communication)
    for ring_step in range(num_rings - 1):
        sends = []
        for ring_id in range(num_rings):
            src_ring = ring_id
            dst_ring = (ring_id + 1) % num_rings
            
            # Send data from first rank of each ring to corresponding rank in next ring
            for rank_offset in range(ranks_per_ring):
                src_rank = src_ring * ranks_per_ring + rank_offset
                dst_rank = dst_ring * ranks_per_ring + rank_offset
                
                # Send chunks from other rings
                for chunk_id in range(num_ranks):
                    if chunk_id // ranks_per_ring != src_ring:  # Only chunks from other rings
                        continue
                    chunk_to_send = chunk_id
                    sends.append([chunk_to_send, src_rank, dst_rank])
        
        if sends:  # Only add step if there are sends
            steps.append(Step(rounds=1, sends=sends))
    
    # Create instance
    instance = Instance(steps=len(steps), chunks=num_ranks)
    
    # Create algorithm
    algorithm = Algorithm(
        name=f'multi_ring_allgather_{num_ranks}rank_{num_rings}rings',
        collective=collective,
        topology=topology,
        instance=instance,
        steps=steps,
        input_map=input_map,
        output_map=output_map
    )
    
    return algorithm

def run_algorithm_demo(algorithm, demo_name):
    """
    Run a single algorithm demonstration
    """
    print(f"\n=== {demo_name} ===")
    print(f"Algorithm: {algorithm.name}")
    print(f"Topology: {algorithm.topology.name}")
    print(f"Ranks: {len(algorithm.input_map)}")
    print(f"Steps: {len(algorithm.steps)}")
    
    # Serialize to JSON
    json_file = os.path.join(os.path.dirname(__file__), f'{algorithm.name}.json')
    save_hccl_object(algorithm, json_file)
    print(f"JSON: {os.path.basename(json_file)}")
    
    # Generate XML
    xml_content = ncclize(algorithm, pretty_print=True)
    xml_file = os.path.join(os.path.dirname(__file__), f'{algorithm.name}.xml')
    with open(xml_file, 'w') as f:
        f.write(xml_content)
    print(f"XML: {os.path.basename(xml_file)}")
    
    # Generate template files
    template_dir = os.path.join(os.path.dirname(__file__), f'{algorithm.name}-template')
    os.makedirs(template_dir, exist_ok=True)
    
    template_generator = HcclAllGatherTemplateGenerator(xml_file)
    generated_files = template_generator.generate_template_files(template_dir)
    print(f"Templates: {len(generated_files)} files in {os.path.basename(template_dir)}/")
    
    return {
        'algorithm': algorithm,
        'json_file': json_file,
        'xml_file': xml_file,
        'template_dir': template_dir,
        'generated_files': generated_files
    }

def main():
    """
    Enhanced tutorial demonstrating multiple AllGather algorithms
    """
    print("=== HCCLang Enhanced AllGather Tutorial ===")
    print("Demonstrating multiple AllGather algorithms with different configurations")
    
    results = []
    
    # Demo 1: Single Ring 4 ranks
    try:
        algorithm = create_single_ring_allgather(4)
        result = run_algorithm_demo(algorithm, "Single Ring 4 Ranks")
        results.append(result)
    except Exception as e:
        print(f"Error in 4-rank demo: {e}")
    
    # Demo 2: Single Ring 8 ranks
    try:
        algorithm = create_single_ring_allgather(8)
        result = run_algorithm_demo(algorithm, "Single Ring 8 Ranks")
        results.append(result)
    except Exception as e:
        print(f"Error in 8-rank demo: {e}")
    
    # Demo 3: Single Ring 16 ranks  
    try:
        algorithm = create_single_ring_allgather(16)
        result = run_algorithm_demo(algorithm, "Single Ring 16 Ranks")
        results.append(result)
    except Exception as e:
        print(f"Error in 16-rank demo: {e}")
    
    # Demo 4: Multi-Ring 8 ranks, 2 rings (if multi_ring topology is available)
    # try:
    #     algorithm = create_multi_ring_allgather(8, 2)
    #     result = run_algorithm_demo(algorithm, "Multi-Ring 8 Ranks (2 Rings)")
    #     results.append(result)
    # except Exception as e:
    #     print(f"Multi-ring demo skipped: {e}")
    #     # Fallback to simplified multi-ring using regular ring topology
    #     try:
    #         algorithm = create_single_ring_allgather(8)
    #         algorithm.name = 'multi_ring_allgather_8rank_2rings_simplified'
    #         result = run_algorithm_demo(algorithm, "Simplified Multi-Ring 8 Ranks")
    #         results.append(result)
    #     except Exception as e2:
    #         print(f"Error in simplified multi-ring demo: {e2}")
    
    print(f"\n=== Tutorial Summary ===")
    print(f"Successfully generated {len(results)} algorithm variants:")
    
    for i, result in enumerate(results, 1):
        algo = result['algorithm']
        print(f"{i}. {algo.name}")
        print(f"   - Ranks: {len(algo.input_map)}, Steps: {len(algo.steps)}")
        print(f"   - Files: JSON, XML, 5 template files")
    
    print(f"\n=== Algorithm Comparison ===")
    print("| Algorithm | Ranks | Steps | Complexity | Scalability |")
    print("|-----------|-------|-------|------------|-------------|")
    
    for result in results:
        algo = result['algorithm']
        ranks = len(algo.input_map)
        steps = len(algo.steps)
        complexity = f"O({steps})"
        scalability = "Good" if steps < ranks else "Moderate"
        print(f"| {algo.name[:20]} | {ranks} | {steps} | {complexity} | {scalability} |")
    
    # print(f"\n=== Key Features Demonstrated ===")
    # print("✅ HCCL serialization format")
    # print("✅ Configurable rank counts (4, 8, 16)")
    # print("✅ Single-ring and multi-ring topologies")
    # print("✅ XML-driven template generation")
    # print("✅ Complete HCCL integration pipeline")
    # print("✅ Scalable algorithm comparison")
    
    return results

if __name__ == "__main__":
    main()