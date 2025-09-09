#!/usr/bin/env python3

"""
HCCLize - DSL to HCCL C++ Code Generator

This module provides the core functionality to transpile HCCLang DSL algorithms
into HCCL-compatible C++ code using Jinja2 templates.

Based on:
- HCCL_API_REFERENCE.md: Core HCCL API documentation
- DSL_HCCL_MAPPING.md: DSL to HCCL mapping specification
- docs/single-ring-allgather-impl/: Reference implementation structure
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re

# Add the parent directory to the path to import HCCLang modules
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from jinja2 import Environment, FileSystemLoader, select_autoescape
except ImportError:
    print("Error: jinja2 is required. Install with: pip install jinja2")
    sys.exit(1)

from hcclang.language import HCCLProgram
from hcclang.language.ir import Program, Gpu, Threadblock, Op, ChunkRef, Buffer as BufferEnum
from hcclang.topologies import generic


class CollectiveType(Enum):
    """Supported collective communication types"""
    ALLGATHER = "allgather"
    ALLREDUCE = "allreduce"
    ALLTOALL = "alltoall"
    BROADCAST = "broadcast"
    REDUCE = "reduce"
    REDUCESCATTER = "reducescatter"


class TopologyType(Enum):
    """Supported network topologies"""
    RING = "ring"
    TREE = "tree"
    MESH = "mesh"
    MULTI_RING = "multi_ring"
    HIERARCHICAL_RING = "hierarchical_ring"


@dataclass
class HcclCodeGenConfig:
    """Configuration for HCCL code generation"""
    collective: CollectiveType
    topology: TopologyType
    output_dir: str
    template_dir: str
    algorithm_name: str
    num_ranks: int
    num_steps: int
    
    # Template variable mappings
    class_name: str = field(init=False)
    guard_name: str = field(init=False)
    collective_name_camel_case: str = field(init=False)
    collective_name_upper: str = field(init=False)
    collective_name_lower: str = field(init=False)
    topo_name: str = field(init=False)
    topo_name_upper: str = field(init=False)
    topo_name_camel_case: str = field(init=False)
    executor_header_file: str = field(init=False)
    collective_base_name: str = field(init=False)
    comm_tag: str = field(init=False)
    
    def __post_init__(self):
        """Initialize derived fields"""
        collective_name = self.collective.value
        
        # Generate class and file names based on topology (no longer use algorithm name)
        self.collective_name_camel_case = self._to_camel_case(collective_name)
        self.collective_name_upper = collective_name.upper()
        self.collective_name_lower = collective_name.lower()
        
        # Use topology for naming
        topology_type = self.topology.value
        self.topo_name = topology_type.lower()
        self.topo_name_upper = topology_type.upper()
        self.topo_name_camel_case = self._to_camel_case(topology_type)
        
        self.class_name = f"{self.collective_name_camel_case}{self.topo_name_camel_case}"
        self.guard_name = f"{self.collective_name_upper}_{self.topo_name_upper}_H"
        self.executor_header_file = f"coll_{self.collective_name_lower}_{self.topo_name}_executor.h"
        self.collective_base_name = f"{self.collective_name_lower}"
        
        # Communication tag mapping
        comm_tag_map = {
            TopologyType.RING: "RING_INNER",
            TopologyType.TREE: "TREE",
            TopologyType.MESH: "MESH"
        }
        self.comm_tag = comm_tag_map.get(self.topology, "RING_INNER")
    
    # Deleted: _extract_algorithm_type - incorrect approach based on naming instead of DSL semantics
    
    def _to_camel_case(self, snake_str: str) -> str:
        """Convert snake_case to CamelCase"""
        components = snake_str.split('_')
        return ''.join(word.capitalize() for word in components)


class DSLToHcclTranspiler:
    """
    Main transpiler class that converts DSL IR to HCCL C++ code
    """
    
    def __init__(self, config: HcclCodeGenConfig):
        self.config = config
        self.jinja_env = self._setup_jinja_environment()
        
    def _setup_jinja_environment(self) -> Environment:
        """Setup Jinja2 environment with templates"""
        # Path to the templates directory relative to this script's location
        template_dir = Path(__file__).parent.absolute() / "templates"
        if not template_dir.exists():
            raise FileNotFoundError(f"Template directory not found: {template_dir}")
        
        env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=select_autoescape(['html', 'xml']),
            trim_blocks=True,
            lstrip_blocks=True
        )
        return env
    
    def transpile_program(self, program: Program) -> Dict[str, str]:
        """
        Transpile a complete DSL program to HCCL C++ files
        
        Args:
            program: The DSL program to transpile
            
        Returns:
            Dictionary mapping file types to generated file paths
        """
        # Analyze the program to extract communication patterns
        comm_analysis = self._analyze_communication_pattern(program)
        
        # Generate algorithm steps code
        algorithm_steps = self._generate_algorithm_steps(program, comm_analysis)
        
        # Generate executor orchestration code (separate from algorithm logic)
        executor_orchestration = self._generate_executor_orchestration(comm_analysis)
        
        # Prepare template variables
        template_vars = self._prepare_template_variables(comm_analysis, algorithm_steps, executor_orchestration)
        
        # Generate all files
        generated_files = {}
        generated_files['alg_header'] = self._generate_algorithm_header(template_vars)
        generated_files['alg_source'] = self._generate_algorithm_source(template_vars)
        generated_files['executor_header'] = self._generate_executor_header(template_vars)
        generated_files['executor_source'] = self._generate_executor_source(template_vars)
        
        return generated_files
    
    def _analyze_communication_pattern(self, program: Program) -> Dict[str, Any]:
        """Analyze DSL program to extract communication patterns and algorithm structure"""
        print(f"  DEBUG: _analyze_communication_pattern called with {len(program.gpus)} gpus")
        analysis = {
            'total_steps': 0,
            'operations_per_step': {},
            'communication_pairs': set(),
            'buffer_usage': {},
            'max_rank': 0,
            'num_rings': 1,
            'nodes_per_level': 4,
            'is_multi_ring': False,
            'is_hierarchical': False,
            'ranks_per_ring': 0,
            'ring_structure': [],
            'communication_phases': [],
            # Additional fields for executor generation
            'requires_zero_copy': False,
            'requires_data_splitting': True,  # Most algorithms need data splitting support
            'requires_slice_preparation': True,  # Most algorithms need slice preparation
            'algorithm_complexity': 'single_level',  # single_level, hierarchical, multi_ring
            'communication_pattern': 'point_to_point'  # point_to_point, collective, broadcast
        }
        
        # Analyze actual DSL program structure
        for gpu in program.gpus:
            analysis['max_rank'] = max(analysis['max_rank'], gpu.rank)
            analysis['buffer_usage'][gpu.rank] = {
                'input_chunks': gpu.input_chunks,
                'output_chunks': gpu.output_chunks,
                'scratch_chunks': gpu.scratch_size()
            }
            
            for tb in gpu.threadblocks:
                for op in tb.ops:
                    step_id = op.step
                    analysis['total_steps'] = max(analysis['total_steps'], step_id + 1)
                    
                    if step_id not in analysis['operations_per_step']:
                        analysis['operations_per_step'][step_id] = []
                    analysis['operations_per_step'][step_id].append(op)
                    
                    # Debug: Print all operations to understand the structure (commented out for clean output)
                    # DEBUG: print(f"    DEBUG: Op on GPU {gpu.rank}, step {step_id}: inst={op.inst}, type={type(op.inst)}")
                    
                    # Extract communication pairs - handle enum instruction types
                    inst_str = str(op.inst).lower() if hasattr(op.inst, 'value') else str(op.inst).lower()
                    
                    if 'send' in inst_str:
                        peer_rank = self._extract_peer_rank(op)
                        if peer_rank is not None:
                            analysis['communication_pairs'].add((gpu.rank, peer_rank))
                            # DEBUG: print(f"    DEBUG: Added send communication pair: ({gpu.rank}, {peer_rank})")
                    elif 'recv' in inst_str:
                        peer_rank = self._extract_peer_rank(op)
                        if peer_rank is not None:
                            analysis['communication_pairs'].add((peer_rank, gpu.rank))
                            # print(f"    DEBUG: Added recv communication pair: ({peer_rank}, {gpu.rank})")
                    elif 'copy' in inst_str:
                        # For ring algorithms, copy operations represent inter-rank data transfer
                        if hasattr(op, 'src') and hasattr(op, 'dst'):
                            src_rank = getattr(op.src, 'rank', None)
                            dst_rank = getattr(op.dst, 'rank', None)
                            # Ring copy operations: from src_rank to current gpu.rank
                            if src_rank is not None and src_rank != gpu.rank:
                                analysis['communication_pairs'].add((src_rank, gpu.rank))
                    
                    # Handle recv_copy_send compound operations - these are ring-specific
                    if 'recv_copy_send' in inst_str:
                        # This suggests a ring pattern where each rank receives, copies, and forwards
                        # For 8-rank ring: rank i receives from (i-1)%8 and sends to (i+1)%8
                        prev_rank = (gpu.rank - 1 + 8) % 8  # Assuming 8 ranks for now
                        next_rank = (gpu.rank + 1) % 8
                        analysis['communication_pairs'].add((prev_rank, gpu.rank))  # receive from prev
                        analysis['communication_pairs'].add((gpu.rank, next_rank))  # send to next
        
        # Set num_steps based on total_steps for compatibility with pattern detection
        analysis['num_steps'] = analysis['total_steps']
        
        # Enhanced analysis based on program structure
        total_ranks = analysis['max_rank'] + 1
        
        # DSL-based analysis will determine these parameters from communication patterns
        
        # Calculate derived parameters
        if analysis['is_multi_ring'] and analysis['num_rings'] > 1:
            analysis['ranks_per_ring'] = total_ranks // analysis['num_rings']
            
            # Build ring structure
            for ring_id in range(analysis['num_rings']):
                ring_start = ring_id * analysis['ranks_per_ring']
                ring_end = ring_start + analysis['ranks_per_ring']
                analysis['ring_structure'].append(list(range(ring_start, ring_end)))
        
        # Initialize pattern-related fields before detection
        print(f"  DEBUG: Initializing pattern fields")
        analysis['pattern'] = 'NOT_SET'
        analysis['topology_type'] = 'NOT_SET' 
        analysis['communication_pattern'] = 'point_to_point'  # Keep original value initially
        analysis['peer_calculation'] = 'NOT_SET'
        
        # Enhanced pattern detection based on communication structure
        print(f"  DEBUG: About to call _detect_algorithm_pattern_from_structure")
        try:
            detected_pattern = self._detect_algorithm_pattern_from_structure(analysis, len(program.gpus), program)
            analysis.update(detected_pattern)
            
            # Debug: print the detected pattern to verify it's working
            print(f"  - Pattern detection result: {detected_pattern}")
            print(f"  - Updated analysis pattern: {analysis.get('pattern', 'NOT_SET')}")
        except Exception as e:
            print(f"  ERROR in pattern detection: {e}")
            # Set default pattern values
            analysis['pattern'] = 'generic'
            analysis['topology_type'] = 'unknown'  
            analysis['communication_pattern'] = 'unknown'
            analysis['peer_calculation'] = 'unknown'
        
        # Analyze communication phases
        analysis['communication_phases'] = self._analyze_communication_phases(analysis)
        
        return analysis
    
    def _analyze_communication_phases(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze communication phases from the algorithm structure"""
        phases = []
        
        if analysis['is_multi_ring']:
            # Phase 1: Intra-ring communication
            phases.append({
                'phase': 'intra_ring',
                'description': 'Intra-ring AllGather within each ring',
                'steps': analysis['ranks_per_ring'] - 1 if analysis['ranks_per_ring'] > 0 else 0,
                'parallel_rings': analysis['num_rings']
            })
            
            # Phase 2: Inter-ring communication (if multiple rings)
            if analysis['num_rings'] > 1:
                phases.append({
                    'phase': 'inter_ring',
                    'description': 'Inter-ring data exchange',
                    'steps': analysis['num_rings'] - 1,
                    'parallel_rings': 1
                })
        elif analysis['is_hierarchical']:
            # Phase 1: Intra-level communication
            phases.append({
                'phase': 'intra_level',
                'description': 'Intra-level AllGather within each level',
                'steps': analysis['nodes_per_level'] - 1,
                'parallel_levels': analysis['max_rank'] + 1 // analysis['nodes_per_level']
            })
            
            # Phase 2: Inter-level communication
            phases.append({
                'phase': 'inter_level',
                'description': 'Inter-level data exchange',
                'steps': (analysis['max_rank'] + 1) // analysis['nodes_per_level'] - 1,
                'parallel_levels': 1
            })
        else:
            # Single-ring communication
            phases.append({
                'phase': 'single_ring',
                'description': 'Single-ring AllGather',
                'steps': analysis['max_rank'],
                'parallel_rings': 1
            })
        
        return phases
    
    def _extract_peer_rank(self, op: Op) -> Optional[int]:
        """Extract peer rank from operation (topology-specific)"""
        try:
            # Try to extract peer rank from operation attributes
            if hasattr(op, 'src') and hasattr(op.src, 'rank'):
                return op.src.rank
            elif hasattr(op, 'dst') and hasattr(op.dst, 'rank'):
                return op.dst.rank
            elif hasattr(op, 'peer_rank'):
                return op.peer_rank
            
            # If no explicit peer rank, try to infer from topology
            if self.config.topology == TopologyType.RING:
                # For ring topology, peer is typically (rank \u00b1 1) % num_ranks
                # This would need the current rank context which we don't have here
                # Return None to indicate we couldn't extract it
                # Ring algorithm uses copy operations with rank information
                if op.inst == 'copy' and hasattr(op, 'src') and hasattr(op, 'dst'):
                    # Try to extract from src or dst buffer rank information
                    if hasattr(op.src, 'rank') and op.src.rank is not None:
                        return op.src.rank
                    elif hasattr(op.dst, 'rank') and op.dst.rank is not None:
                        return op.dst.rank
                return None
            
        except Exception:
            # If extraction fails, return None
            pass
        
        return None
    
    def _generate_algorithm_steps(self, program: Program, comm_analysis: Dict[str, Any]) -> str:
        """Generate the core algorithm steps code from DSL operations"""
        return self.transpile_dsl_to_cpp(program, comm_analysis)

    def transpile_dsl_to_cpp(self, program: Program, comm_analysis: Dict[str, Any] = None) -> str:
        """Transpiles the hcclang Program into a C++ string."""
        cpp_code = []
        indent_level = 1

        # Initial setup
        cpp_code.append(self._indent("u32 unitSize = DataUnitSize(dataType_);", indent_level))
        cpp_code.append(self._indent("if (unitSize == 0) {", indent_level))
        cpp_code.append(self._indent(f'HCCL_ERROR("[{self.config.class_name}][RunAsync]unitSize is zero");', indent_level + 1))
        cpp_code.append(self._indent("return HCCL_E_INTERNAL;", indent_level + 1))
        cpp_code.append(self._indent("}", indent_level))
        cpp_code.append(self._indent("u64 sliceSize = count_ * unitSize;", indent_level))
        cpp_code.append("")

        # Process all operations from DSL to generate complete algorithm
        if program.gpus:
            # Use provided communication analysis if available, otherwise analyze
            if comm_analysis is None:
                algorithm_analysis = self._analyze_algorithm_patterns(program)
            else:
                algorithm_analysis = comm_analysis
                print(f"  DEBUG transpile_dsl_to_cpp: Using provided comm_analysis with pattern: {algorithm_analysis.get('pattern', 'NOT_SET')}")
            
            # Generate algorithm-specific code based on detected patterns
            print(f"  DEBUG: algorithm_analysis keys: {list(algorithm_analysis.keys())}")
            print(f"  DEBUG: algorithm_analysis.get('pattern'): {algorithm_analysis.get('pattern', 'KEY_NOT_FOUND')}")
            if algorithm_analysis.get('pattern', 'NOT_SET') == 'ring':
                cpp_code.extend(self._generate_ring_algorithm(program, algorithm_analysis, indent_level))
            elif algorithm_analysis.get('pattern', 'NOT_SET') == 'recursive_doubling':
                cpp_code.extend(self._generate_recursive_doubling_algorithm(program, algorithm_analysis, indent_level))
            elif algorithm_analysis.get('pattern', 'NOT_SET') == 'mesh' or self.config.topology == TopologyType.MESH:
                # Generate algorithm based on collective type
                if self.config.collective == CollectiveType.ALLTOALL:
                    cpp_code.extend(self._generate_alltoall_algorithm_from_dsl(program, algorithm_analysis, indent_level))
                else:
                    # AllGather or other collectives use mesh pattern
                    cpp_code.extend(self._generate_mesh_algorithm_from_dsl(program, algorithm_analysis, indent_level))
            else:
                # Use generic DSL algorithm for unknown patterns
                cpp_code.extend(self._generate_generic_dsl_algorithm(program, indent_level))
        else:
            # Fallback to generic algorithm generation if no DSL ops
            cpp_code.append(self._indent("// No DSL operations found, using generic algorithm", indent_level))
            cpp_code.append(self._indent("CHK_RET(linkLeft_->TxAck(stream_));", indent_level))
            cpp_code.append(self._indent("CHK_RET(linkRight_->RxAck(stream_));", indent_level))
            cpp_code.append(self._indent("// Generic algorithm implementation would go here", indent_level))

        return "\n".join(cpp_code)

    def _analyze_algorithm_patterns(self, program: Program) -> Dict[str, Any]:
        """Analyze DSL operations to determine algorithm communication patterns"""
        analysis = {
            'pattern': 'generic',
            'topology_type': 'unknown',
            'communication_pattern': 'unknown',
            'peer_calculation': 'unknown',
            'operations_per_step': {},
            'total_operations': 0,
            'peer_expressions': [],
            'communication_phases': [],
            'xor_patterns': [],
            'arithmetic_patterns': [],
            'num_steps': 0,
            'communication_pairs': set()
        }
        
        # Collect all operations and analyze peer patterns
        all_operations = {}
        communication_pairs = set()
        peer_expressions = []
        
        for gpu in program.gpus:
            for tb in gpu.threadblocks:
                for op in tb.ops:
                    step_id = op.step
                    if step_id not in all_operations:
                        all_operations[step_id] = []
                    all_operations[step_id].append(op)
                    analysis['total_operations'] += 1
                    
                    # Extract peer calculation patterns from operations
                    peer_expr = self._extract_peer_expression(op)
                    if peer_expr:
                        peer_expressions.append(peer_expr)
                        analysis['peer_expressions'].append(peer_expr)
                    
                    # Analyze communication pairs
                    if hasattr(op, 'dst') and hasattr(op.dst, 'rank'):
                        peer_rank = op.dst.rank
                        communication_pairs.add((gpu.rank, peer_rank))
        
        analysis['operations_per_step'] = all_operations
        # Calculate correct number of steps: max step + 1 (since steps are 0-indexed)
        print(f"  DEBUG: all_operations keys: {list(all_operations.keys()) if all_operations else 'EMPTY'}")
        try:
            analysis['num_steps'] = max(all_operations.keys()) + 1 if all_operations else 0
            print(f"  DEBUG: num_steps calculated successfully: {analysis['num_steps']}")
        except Exception as e:
            print(f"  ERROR calculating num_steps: {e}")
            analysis['num_steps'] = 0
        analysis['communication_pairs'] = communication_pairs
        
        print(f"  DEBUG: Reached step analysis section, num_steps={analysis['num_steps']}")
        
        # This code is now moved to the correct location before the main return
        
        return analysis
    
    def _detect_algorithm_pattern_from_structure(self, analysis: Dict[str, Any], num_ranks: int, program: Program) -> Dict[str, str]:
        """Detect algorithm pattern based on communication structure and step analysis"""
        communication_pairs = analysis['communication_pairs']
        num_steps = analysis['num_steps']
        operations_per_step = analysis['operations_per_step']
        
        # First, analyze DSL loop structure for better pattern detection
        loop_analysis = self._analyze_dsl_loop_structure(program)
        
        # DSL-based pattern detection prioritizing loop analysis results
        
        # Check for Recursive Doubling from loop analysis first
        if loop_analysis.get('loop_type') == 'recursive_doubling':
            return {
                'pattern': 'recursive_doubling',
                'topology_type': 'fully_connected',
                'communication_pattern': 'all_to_all',
                'peer_calculation': 'xor_distance'
            }
        
        # Check for Recursive Doubling characteristics (most specific)
        # Enhanced detection: check both communication patterns and DSL source patterns
        if (self._is_recursive_doubling_pattern(communication_pairs, num_ranks, num_steps) or 
            self._has_recursive_doubling_dsl_patterns(program)):
            return {
                'pattern': 'recursive_doubling',
                'topology_type': 'fully_connected',
                'communication_pattern': 'all_to_all',
                'peer_calculation': 'xor_distance'
            }
        
        # Check for Ring from loop analysis
        if loop_analysis.get('loop_type') == 'ring':
            return {
                'pattern': 'ring',
                'topology_type': 'ring',
                'communication_pattern': 'neighbor',
                'peer_calculation': 'sequential'
            }
        
        # Check for Mesh/Fully Connected characteristics (before ring)
        if self._is_mesh_communication_pattern(communication_pairs, num_ranks, num_steps):
            return {
                'pattern': 'mesh',
                'topology_type': 'fully_connected',
                'communication_pattern': 'all_to_all',
                'peer_calculation': 'round_robin'
            }
        
        # Check for Ring algorithm characteristics (least specific)
        if self._is_ring_communication_pattern(communication_pairs, num_ranks, num_steps):
            return {
                'pattern': 'ring',
                'topology_type': 'ring',
                'communication_pattern': 'neighbor',
                'peer_calculation': 'sequential'
            }
        
        # Fallback to generic pattern
        return {
            'pattern': 'generic',
            'topology_type': 'unknown',
            'communication_pattern': 'mixed',
            'peer_calculation': 'unknown'
        }
    
    def _is_ring_communication_pattern(self, communication_pairs: set, num_ranks: int, num_steps: int) -> bool:
        """Check if communication pattern matches ring algorithm characteristics"""
        # Ring algorithm characteristics:
        # 1. Each rank communicates with its neighbors: (rank ± 1) % num_ranks
        # 2. Pattern should include both forward and backward ring connections
        # Note: Relaxed step count check since DSL may merge operations into fewer steps
        
        if not communication_pairs or num_ranks < 2:
            return False
        
        # Check if communication follows ring pattern: each rank -> (rank + 1) % num_ranks
        expected_forward_pairs = set()
        expected_backward_pairs = set()
        for rank in range(num_ranks):
            next_rank = (rank + 1) % num_ranks
            prev_rank = (rank - 1 + num_ranks) % num_ranks
            expected_forward_pairs.add((rank, next_rank))
            expected_backward_pairs.add((prev_rank, rank))
        
        # Combine expected patterns
        expected_ring_pairs = expected_forward_pairs.union(expected_backward_pairs)
        
        # Check if at least 60% of communication pairs match ring pattern
        # (relaxed from 80% to handle DSL variations)
        matching_pairs = communication_pairs.intersection(expected_ring_pairs)
        ring_pattern_ratio = len(matching_pairs) / len(expected_ring_pairs) if expected_ring_pairs else 0
        
        # print(f"    DEBUG: Ring pattern check - expected: {len(expected_ring_pairs)}, matching: {len(matching_pairs)}, ratio: {ring_pattern_ratio:.2f}")
        
        return ring_pattern_ratio >= 0.6
    
    def _is_mesh_communication_pattern(self, communication_pairs: set, num_ranks: int, num_steps: int) -> bool:
        """Check if communication pattern matches mesh/fully connected algorithm characteristics"""
        # Mesh algorithm characteristics:
        # 1. Each rank can communicate with all other ranks
        # 2. All-to-all communication pattern 
        # 3. Number of steps typically is (num_ranks - 1) for AllGather
        # 4. Communication pairs should include many different rank connections
        
        if not communication_pairs or num_ranks < 2:
            return False
            
        # Calculate expected communication diversity for mesh
        # In mesh, each rank should communicate with multiple different ranks
        unique_sources = set()
        unique_destinations = set()
        for src, dst in communication_pairs:
            unique_sources.add(src)
            unique_destinations.add(dst)
            
        # Mesh pattern indicators:
        # 1. High communication diversity (many different peer connections)
        # 2. Not following strict neighbor pattern (like ring)
        # 3. Steps close to (num_ranks - 1) for AllGather mesh
        
        communication_diversity_ratio = len(communication_pairs) / (num_ranks * (num_ranks - 1)) if num_ranks > 1 else 0
        participating_ranks_ratio = len(unique_sources) / num_ranks if num_ranks > 0 else 0
        
        # Check if this is mesh pattern based on communication complexity:
        # - High communication diversity (many different connections, not just neighbors)
        # - Many ranks participating (most/all ranks involved)
        # - Complex communication pattern (not simple ring or doubling)
        
        is_high_diversity = communication_diversity_ratio > 0.3  # More diverse than simple patterns
        is_many_participants = participating_ranks_ratio > 0.6   # Most ranks participate
        
        # For mesh pattern, we expect complex all-to-all communication
        # Check if communication pattern is more complex than simple ring
        total_possible_pairs = num_ranks * (num_ranks - 1)
        is_complex_pattern = len(communication_pairs) > num_ranks * 2  # More than simple ring pairs
        
        # Debug mesh pattern detection
        # print(f"DEBUG Mesh pattern: diversity={communication_diversity_ratio:.2f}, participants={participating_ranks_ratio:.2f}")
        # print(f"DEBUG Mesh pattern: pairs={len(communication_pairs)}/{total_possible_pairs}, complex={is_complex_pattern}")
        
        return is_high_diversity and is_many_participants and is_complex_pattern
    
    def _is_recursive_doubling_pattern(self, communication_pairs: set, num_ranks: int, num_steps: int) -> bool:
        """Check if communication pattern matches recursive doubling characteristics"""
        # Recursive doubling characteristics:
        # 1. Number of steps = log2(num_ranks) 
        # 2. Each rank communicates with XOR-distance peers
        # 3. Communication pattern is all-to-all with specific XOR structure
        
        import math
        expected_steps = int(math.log2(num_ranks)) if num_ranks > 0 and (num_ranks & (num_ranks - 1)) == 0 else -1
        
        if expected_steps == -1 or num_steps != expected_steps:
            return False
        
        # Check if communication follows XOR pattern
        # In recursive doubling, each rank communicates with rank ^ (1 << step) for each step
        expected_xor_pairs = set()
        for step in range(expected_steps):
            distance = 1 << step
            for rank in range(num_ranks):
                peer = rank ^ distance
                if peer < num_ranks:
                    expected_xor_pairs.add((rank, peer))
                    expected_xor_pairs.add((peer, rank))  # Bidirectional
        
        # Check if communication pairs match XOR pattern
        matching_pairs = communication_pairs.intersection(expected_xor_pairs)
        xor_pattern_ratio = len(matching_pairs) / len(expected_xor_pairs) if expected_xor_pairs else 0
        
        return xor_pattern_ratio >= 0.6
    
    def _has_recursive_doubling_dsl_patterns(self, program: Program) -> bool:
        """Check if DSL program contains recursive doubling patterns (XOR operations, doubling)"""
        try:
            # Extract all operations from the DSL program
            operations = self._extract_dsl_operations(program)
            
            # Look for XOR patterns in peer calculations
            xor_patterns_found = 0
            doubling_patterns_found = 0
            
            for op_metadata in operations:
                if isinstance(op_metadata, dict):
                    # Check peer expressions for XOR operations
                    peer_expr = str(op_metadata.get('peer_expression', '')).lower()
                    if '^' in peer_expr or 'xor' in peer_expr:
                        xor_patterns_found += 1
                    
                    # Check for doubling patterns (count *= 2, 1 << step, etc.)
                    if ('count' in peer_expr and ('*' in peer_expr or '<<' in peer_expr)) or \
                       ('step' in peer_expr and '<<' in peer_expr) or \
                       ('1 << ' in peer_expr):
                        doubling_patterns_found += 1
            
            # Also check the program structure for loop patterns
            loop_analysis = self._analyze_dsl_loop_structure(program)
            has_xor_loop = loop_analysis.get('peer_calculation_pattern') == 'xor'
            has_doubling_loop = loop_analysis.get('loop_type') == 'recursive_doubling'
            
            # Check program name or related metadata
            program_name = getattr(program, 'name', '')
            has_recursive_name = ('recursive' in program_name.lower() and 'doubling' in program_name.lower())
            
            # Consider it recursive doubling if we have strong indicators
            return (xor_patterns_found > 0 and doubling_patterns_found > 0) or \
                   has_xor_loop or has_doubling_loop or \
                   (xor_patterns_found > 0 and has_recursive_name)
                   
        except Exception as e:
            # If analysis fails, fall back to communication pattern analysis
            return False
    
    def _is_ring_pattern(self, communication_pairs: set, total_ranks: int) -> bool:
        """Check if communication pattern matches ring topology"""
        # Ring pattern: each rank communicates with (rank+1)%n and (rank-1+n)%n
        expected_pairs = set()
        for rank in range(total_ranks):
            next_rank = (rank + 1) % total_ranks
            prev_rank = (rank - 1 + total_ranks) % total_ranks
            expected_pairs.add((rank, next_rank))
            expected_pairs.add((rank, prev_rank))
        
        # Check if communication pairs match ring pattern
        return len(communication_pairs.intersection(expected_pairs)) > len(communication_pairs) * 0.8
    
    def _generate_recursive_doubling_algorithm(self, program: Program, analysis: Dict[str, Any], indent_level: int) -> List[str]:
        """Generate algorithm code based on DSL loop structure analysis"""
        lines = []
        
        # Analyze DSL loop structure
        loop_analysis = self._analyze_dsl_loop_structure(program)
        
        lines.append(self._indent("// Algorithm generated from DSL loop structure analysis", indent_level))
        lines.append(self._indent(f"// Loop analysis: {loop_analysis}", indent_level))
        
        # Debug: Check what operations we have
        operations = self._extract_dsl_operations(program)
        lines.append(self._indent(f"// Debug: Found {len(operations)} DSL operations", indent_level))
        
        if loop_analysis['has_loops'] and loop_analysis['peer_calculation_pattern'] == 'xor':
            # Generate XOR-based loop algorithm
            lines.extend(self._generate_loop_based_algorithm(program, loop_analysis, indent_level))
        else:
            # Fallback to sequence-based generation when no loops detected
            lines.extend(self._generate_from_dsl_operations_sequence(operations, indent_level))
        
        return lines
    
    def _generate_loop_based_algorithm(self, program: Program, loop_analysis: Dict[str, Any], indent_level: int) -> List[str]:
        """Generate loop-based algorithm from DSL structure analysis"""
        lines = []
        
        loop_type = loop_analysis['loop_type']
        iteration_count = loop_analysis['iteration_count']
        peer_pattern = loop_analysis['peer_calculation_pattern']
        
        lines.append(self._indent(f"// Detected {loop_type} algorithm with {peer_pattern} peer calculation", indent_level))
        
        # Generate initialization phase if present
        init_phase = self._extract_initialization_operations(program)
        if init_phase:
            lines.append(self._indent("// Initialization phase", indent_level))
            for op in init_phase:
                op_code = self._generate_operation_code(op['operation'] if isinstance(op, dict) else op)
                if op_code:
                    lines.append(self._indent(op_code, indent_level))
            lines.append(self._indent("", indent_level))
        
        # Generate main communication loop based on detected pattern
        if peer_pattern == 'xor':
            lines.extend(self._generate_xor_communication_loop(program, loop_analysis, indent_level))
        elif peer_pattern == 'arithmetic':
            lines.extend(self._generate_arithmetic_communication_loop(program, loop_analysis, indent_level))
        else:
            lines.extend(self._generate_custom_communication_loop(program, loop_analysis, indent_level))
        
        return lines
    
    def _extract_initialization_operations(self, program: Program) -> List:
        """Extract initialization operations from DSL"""
        operations = self._extract_dsl_operations(program)
        init_ops = []
        
        for op_data in operations:
            if isinstance(op_data, dict) and op_data.get('phase') == 'initialization':
                init_ops.append(op_data)
            elif not isinstance(op_data, dict) and hasattr(op_data, 'inst') and op_data.inst == 'copy':
                # Check if it's initialization copy (input -> output)
                if hasattr(op_data, 'src') and hasattr(op_data, 'dst'):
                    src_buffer = str(op_data.src.buffer).split('.')[-1] if hasattr(op_data.src.buffer, 'value') else str(op_data.src.buffer)
                    dst_buffer = str(op_data.dst.buffer).split('.')[-1] if hasattr(op_data.dst.buffer, 'value') else str(op_data.dst.buffer)
                    if src_buffer == 'input' and dst_buffer == 'output':
                        init_ops.append(op_data)
        
        return init_ops
    
    def _generate_xor_communication_loop(self, program: Program, loop_analysis: Dict[str, Any], indent_level: int) -> List[str]:
        """Generate XOR-based communication loop from DSL operations"""
        lines = []
        
        # Extract the actual XOR expressions from DSL operations
        operations = self._extract_dsl_operations(program)
        comm_operations = [op for op in operations if isinstance(op, dict) and op['instruction'] in ['send', 'recv', 'copy']]
        
        # Analyze the XOR pattern from actual DSL expressions
        xor_expressions = []
        for op in comm_operations:
            peer_expr = op.get('peer_expression', '')
            if peer_expr and ('^' in str(peer_expr) or 'xor' in str(peer_expr).lower()):
                xor_expressions.append(peer_expr)
        
        # Extract loop variable from XOR expressions
        loop_var = self._extract_loop_variable_from_expressions(xor_expressions)
        
        lines.append(self._indent(f"// XOR-based communication loop (detected from DSL)", indent_level))
        
        # Determine loop bounds from DSL analysis
        iteration_count = loop_analysis['iteration_count']
        if iteration_count > 0:
            # Generate loop based on detected iteration pattern
            if loop_var and 'count' in str(loop_var).lower():
                # Pattern like: peer = rank ^ count; count *= 2
                lines.append(self._indent(f"u32 count = 1;", indent_level))
                lines.append(self._indent(f"u32 iteration = 0;", indent_level))
                lines.append(self._indent(f"u32 maxIterations = {iteration_count};", indent_level))
                lines.append(self._indent(f"while (iteration < maxIterations) {{", indent_level))
                
                # Generate XOR peer calculation from DSL
                lines.append(self._indent(f"    u32 peer = rank ^ count;  // From DSL: {xor_expressions[0] if xor_expressions else 'rank ^ count'}", indent_level + 1))
                
            else:
                # Pattern like: peer = rank ^ (1 << step)
                lines.append(self._indent(f"for (u32 step = 0; step < {iteration_count}; step++) {{", indent_level))
                lines.append(self._indent(f"    u32 peer = rank ^ (1 << step);  // XOR pattern from DSL", indent_level + 1))
        else:
            # Fallback - generate generic XOR loop
            lines.append(self._indent(f"for (u32 step = 0; step < log2(rankSize); step++) {{", indent_level))
            lines.append(self._indent(f"    u32 peer = rank ^ (1 << step);  // XOR pattern from DSL", indent_level + 1))
        
        # Generate loop body based on DSL operations
        lines.extend(self._generate_loop_body_from_dsl_operations(program, comm_operations, indent_level + 1))
        
        # Close loop
        if loop_var and 'count' in str(loop_var).lower():
            lines.append(self._indent(f"    count *= 2;", indent_level + 1))
            lines.append(self._indent(f"    iteration++;", indent_level + 1))
            lines.append(self._indent(f"}}", indent_level))
        else:
            lines.append(self._indent(f"}}", indent_level))
        
        return lines
    
    def _extract_loop_variable_from_expressions(self, xor_expressions: List[str]) -> str:
        """Extract loop variable from XOR expressions"""
        for expr in xor_expressions:
            expr_str = str(expr).lower()
            if 'count' in expr_str:
                return 'count'
            elif 'step' in expr_str:
                return 'step'
            elif 'distance' in expr_str:
                return 'distance'
        return 'step'  # default
    
    def _generate_loop_body_from_dsl_operations(self, program: Program, comm_operations: List, indent_level: int) -> List[str]:
        """Generate loop body from DSL communication operations"""
        lines = []
        
        lines.append(self._indent("if (peer >= rankSize) {", indent_level))
        lines.append(self._indent("    continue;  // Skip invalid peers", indent_level))
        lines.append(self._indent("}", indent_level))
        lines.append(self._indent("if (peer >= links.size()) {", indent_level))
        lines.append(self._indent(f"    HCCL_ERROR(\"[{self.config.class_name}][Loop] peer[%u] >= linkSize[%zu]\", peer, links.size());", indent_level))
        lines.append(self._indent("    return HCCL_E_INTERNAL;", indent_level))
        lines.append(self._indent("}", indent_level))
        lines.append(self._indent("CHK_SMART_PTR_NULL(links[peer]);", indent_level))
        lines.append(self._indent("", indent_level))
        
        # Re-extract operations to ensure we have fresh data
        operations = self._extract_dsl_operations(program)
        
        # Filter communication operations directly (handle enum instructions)
        def is_comm_op(op):
            if not isinstance(op, dict):
                return False
            instruction = op.get('instruction')
            if hasattr(instruction, 'value'):
                return instruction.value in ['send', 'recv', 'copy']
            return str(instruction) in ['send', 'recv', 'copy']
            
        actual_comm_ops = [op for op in operations if is_comm_op(op)]
        
        # Use actual_comm_ops instead of passed comm_operations (handle enum instructions)
        def get_instruction_value(op):
            instruction = op.get('instruction')
            if hasattr(instruction, 'value'):
                return instruction.value
            return str(instruction)
            
        send_ops = [op for op in actual_comm_ops if get_instruction_value(op) in ['send', 'copy']]
        recv_ops = [op for op in actual_comm_ops if get_instruction_value(op) in ['recv', 'copy']]
        
        # For recursive doubling, copy operations represent bidirectional communication
        has_communication = send_ops or recv_ops or any(get_instruction_value(op) == 'copy' for op in actual_comm_ops)
        
        if has_communication:
            # Generate asymmetric handshake to avoid deadlock
            lines.append(self._indent("// Asymmetric handshake protocol (rank ID-based ordering to avoid deadlock)", indent_level))
            lines.append(self._indent("if (rank < peer) {", indent_level))
            lines.append(self._indent("    // Lower rank ID: initiate handshake first", indent_level + 1))
            lines.append(self._indent("    CHK_RET(links[peer]->TxAck(stream_));  // Signal: I'm ready to receive", indent_level + 1))
            lines.append(self._indent("    CHK_RET(links[peer]->RxAck(stream_));  // Wait: peer is ready to send", indent_level + 1))
            lines.append(self._indent("} else {", indent_level))
            lines.append(self._indent("    // Higher rank ID: respond to handshake", indent_level + 1))
            lines.append(self._indent("    CHK_RET(links[peer]->RxAck(stream_));  // Wait: peer initiates handshake", indent_level + 1))
            lines.append(self._indent("    CHK_RET(links[peer]->TxAck(stream_));  // Response: I'm ready to send", indent_level + 1))
            lines.append(self._indent("}", indent_level))
            lines.append(self._indent("", indent_level))
            
            # For recursive doubling, generate correct step-based data exchange
            lines.append(self._indent("// Recursive doubling data exchange for step", indent_level))
            lines.append(self._indent("u32 exchangeSize = 1 << step;  // 2^step elements to exchange", indent_level))
            lines.append(self._indent("", indent_level))
            
            # Calculate correct data ranges for recursive doubling
            lines.append(self._indent("// Calculate data ranges based on recursive doubling algorithm", indent_level))
            lines.append(self._indent("u32 myGroupStart = (rank / exchangeSize) * exchangeSize;", indent_level))
            lines.append(self._indent("u32 peerGroupStart = (peer / exchangeSize) * exchangeSize;", indent_level))
            lines.append(self._indent("", indent_level))
            
            # Generate send operations - send what we have in our group
            lines.append(self._indent("// Send data from our group that peer needs", indent_level))
            lines.append(self._indent("for (u32 sendRank = myGroupStart; sendRank < myGroupStart + exchangeSize; sendRank++) {", indent_level))
            lines.append(self._indent("    if (sendRank < rankSize) {", indent_level + 1))
            lines.append(self._indent("        Slice sendSlice = outputSlices[sendRank];", indent_level + 2))
            lines.append(self._indent("        CHK_RET(Tx(links[peer], sendSlice));", indent_level + 2))
            lines.append(self._indent("    }", indent_level + 1))
            lines.append(self._indent("}", indent_level))
            lines.append(self._indent("", indent_level))
            
            # Generate receive operations - receive peer's group data
            lines.append(self._indent("// Receive data from peer's group", indent_level))
            lines.append(self._indent("for (u32 recvRank = peerGroupStart; recvRank < peerGroupStart + exchangeSize; recvRank++) {", indent_level))
            lines.append(self._indent("    if (recvRank < rankSize) {", indent_level + 1))
            lines.append(self._indent("        Slice recvSlice = outputSlices[recvRank];", indent_level + 2))
            lines.append(self._indent("        CHK_RET(Rx(links[peer], recvSlice));", indent_level + 2))
            lines.append(self._indent("    }", indent_level + 1))
            lines.append(self._indent("}", indent_level))
            lines.append(self._indent("", indent_level))
            
            # Generate completion synchronization
            lines.append(self._indent("// Wait for completion", indent_level))
            lines.append(self._indent("CHK_RET(links[peer]->TxWaitDone(stream_));", indent_level))
            lines.append(self._indent("CHK_RET(links[peer]->RxWaitDone(stream_));", indent_level))
        
        return lines
    
    def _generate_executor_constructor_code(self) -> str:
        """Generate executor constructor initialization code"""
        return "    DMAReduceFlag_ = false;"
    
    def _generate_executor_calc_stream_num_code(self, comm_analysis: Dict) -> str:
        """Generate CalcStreamNum method code based on algorithm analysis"""
        lines = []
        
        lines.extend([
            "    u32 totalStreamNum = (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING ? LEVEL0_PLANE_NUM_IN_NPRING_DOUBLE :",
            "        LEVEL0_PLANE_NUM_IN_NPRING_SINGLE);",
            "    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {",
            "        totalStreamNum *= STREAM_NUM_FOR_DMAREDUCE_ONE_RING;",
            "    }",
            "    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB &&",
            "        GetExternalInputEnableRdmaSdmaConcurrent()) {",
            "        totalStreamNum += (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) ? LEVEL0_PLANE_NUM_IN_NPRING_DOUBLE :",
            "        LEVEL0_PLANE_NUM_IN_NPRING_SINGLE;",
            "    }",
            "    streamNum = totalStreamNum - 1;",
            f'    HCCL_INFO("[{self.config.class_name}][CalcStreamNum] tag[%s] streamNum_[%u]",',
            "        tag_.c_str(), streamNum);"
        ])
            
        return '\n'.join(lines)
    
    def _generate_executor_calc_comm_info_code(self, comm_analysis: Dict) -> str:
        """Generate CalcCommInfo method code"""
        lines = [
            "    TransportMemType inputType = TransportMemType::RESERVED;",
            "    TransportMemType outputType = TransportMemType::RESERVED;",
            "    CHK_RET(CalcTransportMemType(inputType, outputType));",
            "    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));"
        ]
        
        if comm_analysis.get('is_multi_ring', False) and comm_analysis.get('num_rings', 1) > 1:
            num_rings = comm_analysis['num_rings']
            lines.append(f"    // Multi-ring algorithm: Create {num_rings} communication planes")
            lines.append(f"    for (u32 ringIndex = 1; ringIndex < {num_rings}; ringIndex++) {{")
            lines.append("        CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport, ringIndex));")
            lines.append("    }")
            
        lines.append("    CHK_RET(CalcLevel1CommInfo(inputType, outputType, opTransport));")
        
        return '\n'.join(lines)
    
    def _generate_executor_calc_level1_comm_info_code(self, comm_analysis: Dict) -> str:
        """Generate CalcLevel1CommInfo method code for inter-server communication"""
        lines = [
            "    bool isSelectAHC = (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC ||",
            "        algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE);",
            "    CommPlane commPlaneLevel1 = isSelectAHC ? COMM_LEVEL1_AHC : COMM_LEVEL1;",
            "    CHK_RET(CheckCommSize(commPlaneLevel1, COMM_INDEX_0 + 1));",
            "    SubCommInfo level1CommInfo = GetSubCommInfo(commPlaneLevel1, COMM_INDEX_0);",
            "",
            "    CommParaInfo commParaLevel1(COMM_LEVEL1, CommType::COMM_TAG_RING_INNER);",
            "    if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {",
            "        commParaLevel1.commType = CommType::COMM_TAG_NONUNIFORM_BRUCK;",
            "    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {",
            "        commParaLevel1.commType = CommType::COMM_TAG_NONUNIFORM_HIERARCHICAL_RING;",
            "    }",
            "    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel1, opTransport[COMM_LEVEL1], inputType, outputType));"
        ]
        return '\n'.join(lines)

    def _generate_executor_calc_level2_comm_info_code(self, comm_analysis: Dict) -> str:
        """Generate CalcLevel2CommInfo method code for inter-superpod communication"""
        lines = [
            "    if( algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC ||",
            "        algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE) {",
            "        HCCL_INFO(\"[" + self.config.class_name + "][CalcLevel2CommInfo] select AHC bypass level2 comm calculate\");",
            "        return HCCL_SUCCESS;",
            "    }",
            "",
            "    CommParaInfo commParaLevel2(COMM_LEVEL2, CommType::COMM_TAG_MAX);",
            "    if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NHR) {",
            "        commParaLevel2.commType = CommType::COMM_TAG_NONUNIFORM_HIERARCHICAL_RING;",
            "        HCCL_INFO(\"[%s]Calc NHRCommInfo\", __func__);",
            "    } else if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NB) {",
            "        commParaLevel2.commType = CommType::COMM_TAG_NONUNIFORM_BRUCK;",
            "        HCCL_INFO(\"[%s]Calc NBCommInfo\", __func__);",
            "    } else {",
            "        commParaLevel2.commType = CommType::COMM_TAG_RING_INNER;",
            "        HCCL_INFO(\"[%s]Calc RingCommInfo\", __func__);",
            "    }",
            "    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel2, opTransport[COMM_LEVEL2], inputType, outputType));"
        ]
        return '\n'.join(lines)

    def _generate_executor_parse_param_code(self) -> str:
        """Generate ParseParam method code"""
        lines = [
            "    tag_ = param.tag;",
            "    root_ = param.root;",
            "    aicpuUnfoldMode_ = param.aicpuUnfoldMode;",
            "    opType_ = param.opType;",
            "    isZeroCopy_ = param.isZeroCopy;"
        ]
        return '\n'.join(lines)

    def _generate_executor_is_data_split_code(self) -> str:
        """Generate IsDataSplitForRdmaSdmaConcurrent method code"""
        lines = [
            "    bool isLargeSize = (curSize >= HCCL_SPLIT_SIZE_INTER_SERVER);",
            "    return GetExternalInputEnableRdmaSdmaConcurrent() && (topoAttr_.serverNum > 1) && isLargeSize;"
        ]
        return '\n'.join(lines)

    def _generate_executor_kernel_run_inter_server_code(self, comm_analysis: Dict) -> str:
        """Generate KernelRunInterServer method code - deprecated, now delegated to KernelRun"""
        lines = [
            f'    HCCL_INFO("[{self.config.class_name}][KernelRunInterServer] Delegating to KernelRun");',
            "    return KernelRun(param, execMem);"
        ]
        return '\n'.join(lines)

    def _generate_executor_kernel_run_intra_server_code(self, comm_analysis: Dict) -> str:
        """Generate KernelRunIntraServer method code - deprecated, now delegated to KernelRun"""
        lines = [
            f'    HCCL_INFO("[{self.config.class_name}][KernelRunIntraServer] Delegating to KernelRun");',
            "    return KernelRun(param, execMem);"
        ]
        return '\n'.join(lines)

    def _generate_executor_orchestrate_code(self, comm_analysis: Dict) -> str:
        """Generate Orchestrate method code for algorithm resource orchestration"""
        lines = [
            f'    HCCL_INFO("[{self.config.class_name}][Orchestrate] Starting algorithm orchestration");',
            "",
            "    // Parse operation parameters",
            "    ParseParam(param);",
            "",
            "    // Calculate communication info and resource requirements",
            "    CHK_RET(CalcCommInfo(algRes.opTransportResponse));",
            "",
            "    // Algorithm orchestration - prepare execution memory",
            "    ExecMem execMem;",
            "    execMem.count = param.DataDes.count;",
            "    execMem.inputMem = algRes.cclInputMem;",
            "    execMem.outputMem = algRes.cclOutputMem;",
            "    execMem.inputPtr = param.inputPtr;",
            "    execMem.outputPtr = param.outputPtr;",
            "",
            "    // Execute main algorithm through KernelRun",
            "    CHK_RET(KernelRun(param, execMem));"
        ]
        return '\n'.join(lines)
    
    def _generate_executor_calc_level0_comm_info_code(self, comm_analysis: Dict) -> str:
        """Generate CalcLevel0CommInfo method code"""
        lines = [
            f"    // {self.config.topo_name_camel_case} topology communication setup",
            f"    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_{self.config.comm_tag});"
        ]
        
        if self.config.topo_name == "ring":
            lines.extend([
                "    commParaLevel0.meshSinglePlane = (topoAttr_.deviceType == DevType::DEV_TYPE_910B) &&",
                "        !topoMatcher_->GetExternalInputHcclDeterministic() &&",
                "        (workflowMode_ != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);"
            ])
            
        lines.append("    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));")
        
        return '\n'.join(lines)
    
    def _generate_executor_calc_transport_mem_type_code(self) -> str:
        """Generate CalcTransportMemType method code"""
        lines = [
            "    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {",
            "        inputType = TransportMemType::CCL_INPUT;",
            "        outputType = TransportMemType::CCL_OUTPUT;",
            "    } else {",
            "        inputType = TransportMemType::PARAM_INPUT;",
            "        outputType = TransportMemType::PARAM_OUTPUT;",
            "    }",
            f'    HCCL_INFO("[{self.config.class_name}][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d]",',
            '        tag_.c_str(), inputType, outputType);'
        ]
        
        return '\n'.join(lines)
    
    def _generate_executor_calc_loop_max_count_code(self, comm_analysis: Dict) -> str:
        """Generate CalcLoopMaxCount method code"""
        lines = [
            "    u64 maxCountPerLoop = cclBuffSize / (topoAttr_.userRankSize * unitSize);",
            "    return maxCountPerLoop;"
        ]
        
        return '\n'.join(lines)
    
    def _generate_executor_additional_interfaces(self, comm_analysis: Dict) -> str:
        """Generate additional interface declarations based on algorithm requirements"""
        interfaces = []
        
        # Add interfaces based on algorithm characteristics
        if comm_analysis.get('requires_zero_copy', False):
            interfaces.extend([
                "    u64 CalcLoopMaxCountZeroCopy(const u32 unitSize, const bool isZeroCopy);",
                "    HcclResult KernelRunInterServer(const OpParam &param, ExecMem &execMem);",
                "    HcclResult KernelRunIntraServer(const OpParam &param, ExecMem &execMem);"
            ])
            
        if comm_analysis.get('requires_data_splitting', False):
            interfaces.extend([
                "    bool IsHugeData(const u64 curSize);",
                "    bool IsSmallData(const u64 size);",
                "    bool IsDataSplitForRdmaSdmaConcurrent(const u64 curSize);"
            ])
            
        if comm_analysis.get('requires_slice_preparation', False):
            interfaces.extend([
                f"    HcclResult Prepare{self.config.collective_name_camel_case}Slice(u32 sliceNum, u64 inputMemSize,",
                "        std::vector<Slice> &dataSegsSlice) const;"
            ])
            
        return '\n'.join(interfaces) if interfaces else ""
    
    def _generate_dsl_algorithm_function(self, comm_analysis: Dict, algorithm_steps: str) -> str:
        """Generate DSL algorithm function - stub for template compatibility"""
        # This method is kept as stub to avoid template errors, returns empty string
        return ""
    
    def _generate_dsl_algorithm_function_declarations(self, comm_analysis: Dict) -> str:
        """Generate DSL algorithm function declarations - stub for template compatibility"""
        # This method is kept as stub to avoid template errors, returns empty string
        return ""
    
    def _generate_dynamic_send_operation(self, send_op: Dict, base_indent: int) -> List[str]:
        """Generate dynamic send operation code from DSL operation"""
        lines = []
        indent_level = base_indent
        
        # Extract buffer information from DSL
        src_buffer = send_op.get('src_buffer', {})
        
        lines.append(self._indent("// Dynamic send based on loop iteration", indent_level))
        lines.append(self._indent("{", indent_level))
        lines.append(self._indent("    u64 srcOffset = slices_[rank].offset;", indent_level))
        lines.append(self._indent("    u64 dataSize = slices_[rank].size;", indent_level))
        lines.append(self._indent("    DeviceMem srcMem = outputMem_.range(srcOffset, dataSize);", indent_level))
        lines.append(self._indent("    CHK_RET(links[peer]->RxAck(stream_));", indent_level))
        lines.append(self._indent("    CHK_RET(links[peer]->TxAsync(UserMemType::OUTPUT_MEM, srcOffset + baseOffset_, srcMem.ptr(), dataSize, stream_));", indent_level))
        lines.append(self._indent("}", indent_level))
        
        return lines
    
    def _generate_dynamic_recv_operation(self, recv_op: Dict, base_indent: int) -> List[str]:
        """Generate dynamic receive operation code from DSL operation"""
        lines = []
        indent_level = base_indent
        
        lines.append(self._indent("// Dynamic receive based on loop iteration", indent_level))
        lines.append(self._indent("{", indent_level))
        lines.append(self._indent("    u64 dstOffset = slices_[peer].offset;", indent_level))
        lines.append(self._indent("    u64 dataSize = slices_[peer].size;", indent_level))
        lines.append(self._indent("    DeviceMem dstMem = outputMem_.range(dstOffset, dataSize);", indent_level))
        lines.append(self._indent("    CHK_RET(links[peer]->RxAsync(UserMemType::OUTPUT_MEM, dstOffset + baseOffset_, dstMem.ptr(), dataSize, stream_));", indent_level))
        lines.append(self._indent("    CHK_RET(links[peer]->DataReceivedAck(stream_));", indent_level))
        lines.append(self._indent("}", indent_level))
        
        return lines
    
    def _generate_arithmetic_communication_loop(self, program: Program, loop_analysis: Dict[str, Any], indent_level: int) -> List[str]:
        """Generate arithmetic-based communication loop for ring algorithms"""
        lines = []
        
        operations = self._extract_dsl_operations(program)
        comm_operations = [op for op in operations if isinstance(op, dict) and op['instruction'] in ['send', 'recv', 'copy']]
        
        lines.append(self._indent("// Ring-based communication loop (detected from DSL)", indent_level))
        iteration_count = loop_analysis['iteration_count']
        
        lines.append(self._indent(f"for (u32 step = 0; step < {iteration_count}; step++) {{", indent_level))
        lines.append(self._indent("    u32 peer = (rank + 1) % rankSize;  // Ring pattern from DSL", indent_level + 1))
        
        # Generate loop body
        lines.extend(self._generate_loop_body_from_dsl_operations(program, comm_operations, indent_level + 1))
        
        lines.append(self._indent("}", indent_level))
        
        return lines
    
    def _generate_custom_communication_loop(self, program: Program, loop_analysis: Dict[str, Any], indent_level: int) -> List[str]:
        """Generate custom communication loop from DSL operations"""
        lines = []
        
        operations = self._extract_dsl_operations(program)
        comm_operations = [op for op in operations if isinstance(op, dict) and op['instruction'] in ['send', 'recv', 'copy']]
        
        lines.append(self._indent("// Custom communication pattern (detected from DSL)", indent_level))
        iteration_count = loop_analysis['iteration_count']
        
        # Generate sequential processing of DSL operations
        lines.extend(self._generate_from_dsl_operations_sequence(operations, indent_level))
        
        return lines
    
    def _get_peer_calculation(self, op: Op) -> str:
        """Generate peer calculation code based on DSL operation analysis"""
        # First priority: Extract exact peer calculation from DSL operation
        if hasattr(op, 'dst') and hasattr(op.dst, 'rank_str'):
            rank_expr = op.dst.rank_str.replace("g.rank", "rank")
            # Detect XOR pattern in DSL expression
            if '^' in rank_expr or 'XOR' in rank_expr.upper():
                return f"u32 peer = {rank_expr};  // XOR pattern detected from DSL"
            else:
                return f"u32 peer = ({rank_expr} + rankSize) % rankSize;  // Arithmetic pattern from DSL"
        elif hasattr(op, 'src') and hasattr(op.src, 'rank_str'):
            rank_expr = op.src.rank_str.replace("g.rank", "rank")
            if '^' in rank_expr or 'XOR' in rank_expr.upper():
                return f"u32 peer = {rank_expr};  // XOR pattern detected from DSL"
            else:
                return f"u32 peer = ({rank_expr} + rankSize) % rankSize;  // Arithmetic pattern from DSL"
        
        # Second priority: Analyze peer from direct rank references
        if hasattr(op, 'dst') and hasattr(op.dst, 'rank') and isinstance(op.dst.rank, int):
            return f"u32 peer = {op.dst.rank};  // Direct peer rank from DSL"
        elif hasattr(op, 'src') and hasattr(op.src, 'rank') and isinstance(op.src.rank, int):
            return f"u32 peer = {op.src.rank};  // Direct peer rank from DSL"
        
        # Third priority: Analyze communication pattern from operation context
        return self._analyze_operation_peer_pattern(op)
    
    def _get_link_selection(self, op: Op, peer_calculation: str) -> str:
        """Generate dynamic link selection code based on algorithm pattern and peer calculation"""
        # Detect algorithm pattern to determine link selection strategy
        algorithm_pattern = self._detect_algorithm_pattern_from_operation(op)
        
        if algorithm_pattern == 'recursive_doubling' or 'xor' in peer_calculation.lower():
            # Recursive doubling uses dynamic link selection based on XOR peer calculation
            return f"""        {peer_calculation}
        // Dynamic link selection for recursive doubling/XOR pattern
        if (peer >= links.size()) {{
            HCCL_ERROR("[{self.config.class_name}][DSL Operation] peer[%u] >= linkSize[%zu]", peer, links.size());
            return HCCL_E_INTERNAL;
        }}
        LINK peerLink = links[peer];  // Direct peer access for all-to-all communication
        CHK_SMART_PTR_NULL(peerLink);"""
        
        elif algorithm_pattern == 'ring' or ('sequential' in peer_calculation.lower() or '+1' in peer_calculation):
            # Ring algorithms use left/right link pattern
            return f"""        {peer_calculation}
        // Ring topology link selection
        LINK peerLink = nullptr;
        if (peer == (rank + 1) % rankSize) {{
            peerLink = linkRight_;  // Send to next rank in ring
        }} else if (peer == (rank - 1 + rankSize) % rankSize) {{
            peerLink = linkLeft_;   // Send to previous rank in ring  
        }} else {{
            // General peer link for non-standard ring communication
            if (peer >= links.size()) {{
                HCCL_ERROR("[{self.config.class_name}][DSL Operation] peer[%u] >= linkSize[%zu]", peer, links.size());
                return HCCL_E_INTERNAL;
            }}
            peerLink = links[peer];
        }}
        CHK_SMART_PTR_NULL(peerLink);"""
        
        else:
            # Generic dynamic link selection
            return f"""        {peer_calculation}
        // Generic dynamic link selection  
        if (peer >= links.size()) {{
            HCCL_ERROR("[{self.config.class_name}][DSL Operation] peer[%u] >= linkSize[%zu]", peer, links.size());
            return HCCL_E_INTERNAL;
        }}
        LINK peerLink = links[peer];
        CHK_SMART_PTR_NULL(peerLink);"""
    
    def _detect_algorithm_pattern_from_operation(self, op: Op) -> str:
        """Detect algorithm pattern from single operation context"""
        # Check peer expression in the operation
        peer_expr = self._extract_peer_expression(op)
        if peer_expr:
            if '^' in peer_expr or 'xor' in peer_expr.lower():
                return 'recursive_doubling'
            elif '+1' in peer_expr or '-1' in peer_expr or 'sequential' in peer_expr.lower():
                return 'ring'
        
        return 'generic'
    
    def _extract_dsl_operations(self, program: Program) -> List:
        """Extract operations from DSL program and analyze loop patterns"""
        operations = []
        if not program.gpus:
            return operations
            
        # Extract operations with enhanced metadata and loop analysis
        for gpu in program.gpus:
            for threadblock in gpu.threadblocks:
                for op in threadblock.ops:
                    # Enrich operation with analysis metadata
                    op_with_metadata = {
                        'operation': op,
                        'gpu_rank': gpu.rank,
                        'threadblock_id': threadblock.threadblock_id if hasattr(threadblock, 'threadblock_id') else 0,
                        'step': op.step,
                        'instruction': op.inst,
                        'phase': self._classify_operation_phase(op),
                        'peer_expression': self._extract_peer_expression(op),
                        'src_buffer': self._extract_buffer_info(getattr(op, 'src', None)),
                        'dst_buffer': self._extract_buffer_info(getattr(op, 'dst', None))
                    }
                    operations.append(op_with_metadata)
        
        # Sort operations by step for proper sequence
        operations.sort(key=lambda x: (x['step'], x['gpu_rank']))
        
        return operations
    
    def _analyze_dsl_loop_structure(self, program: Program) -> Dict[str, Any]:
        """Analyze DSL program to identify loop patterns and iteration logic"""
        loop_analysis = {
            'has_loops': False,
            'loop_type': 'unknown',
            'iteration_variable': None,
            'loop_condition': None,
            'loop_increment': None,
            'loop_body_operations': [],
            'peer_calculation_pattern': None,
            'data_exchange_pattern': None
        }
        
        operations = self._extract_dsl_operations(program)
        if not operations:
            return loop_analysis
        
        # Group operations by step to identify iteration patterns
        step_groups = {}
        for op_metadata in operations:
            step = op_metadata['step']
            if step not in step_groups:
                step_groups[step] = []
            step_groups[step].append(op_metadata)
        
        # Analyze step progression to identify loop patterns
        steps = sorted(step_groups.keys())
        if len(steps) > 1:
            # Check for recurring patterns indicating loops
            loop_pattern = self._detect_loop_pattern_from_steps(step_groups, steps)
            loop_analysis.update(loop_pattern)
        
        return loop_analysis
    
    def _detect_loop_pattern_from_steps(self, step_groups: Dict, steps: List[int]) -> Dict[str, Any]:
        """Detect loop patterns from step analysis"""
        pattern_info = {
            'has_loops': False,
            'loop_type': 'unknown',
            'peer_calculation_pattern': None,
            'iteration_count': 0
        }
        
        # Analyze communication operations across steps (include copy operations for recursive doubling)
        communication_steps = []
        for step in steps:
            # Include copy operations which are used in recursive doubling algorithms
            comm_ops = [op for op in step_groups[step] if self._get_instruction_value(op) in ['send', 'recv', 'copy']]
            if comm_ops:
                communication_steps.append({
                    'step': step,
                    'operations': comm_ops,
                    'peer_expressions': [op['peer_expression'] for op in comm_ops if op['peer_expression']]
                })
        
        # Check if we have many steps with similar communication patterns (indicates loop expansion)
        if len(communication_steps) > 5:  # More than 5 communication steps likely indicates loop expansion
            pattern_info['has_loops'] = True
            pattern_info['iteration_count'] = len(communication_steps)
            
            # Analyze communication pattern across steps to detect recursive doubling
            peer_rank_counts = {}  # Count how many times each peer rank appears
            all_peer_expressions = []
            
            for step_info in communication_steps:
                for op in step_info['operations']:
                    # Extract actual peer rank from operation
                    peer_rank = self._extract_actual_peer_rank(op)
                    if peer_rank is not None:
                        peer_rank_counts[peer_rank] = peer_rank_counts.get(peer_rank, 0) + 1
                    
                    # Collect peer expressions
                    if op['peer_expression']:
                        all_peer_expressions.append(op['peer_expression'])
            
            # Detect recursive doubling pattern: operations should involve many different peers
            # In recursive doubling, each rank communicates with log(n) different peers
            unique_peers = len(peer_rank_counts)
            total_ranks = max(peer_rank_counts.keys()) + 1 if peer_rank_counts else 8
            
            if unique_peers >= 3 and unique_peers >= total_ranks // 2:
                # Many unique peers suggests recursive doubling or full mesh pattern
                pattern_info['loop_type'] = 'recursive_doubling'
                pattern_info['peer_calculation_pattern'] = 'xor'
            else:
                # Check for XOR patterns in expressions (backup check)
                xor_patterns = [p for p in all_peer_expressions if '^' in str(p) or 'xor' in str(p).lower()]
                if xor_patterns:
                    pattern_info['loop_type'] = 'recursive_doubling'
                    pattern_info['peer_calculation_pattern'] = 'xor'
                elif self._is_arithmetic_progression_pattern(all_peer_expressions):
                    pattern_info['loop_type'] = 'ring'
                    pattern_info['peer_calculation_pattern'] = 'arithmetic'
                else:
                    pattern_info['loop_type'] = 'custom'
                    pattern_info['peer_calculation_pattern'] = 'custom'
        elif len(communication_steps) > 1:
            # Standard loop detection for smaller step counts
            pattern_info['has_loops'] = True
            pattern_info['iteration_count'] = len(communication_steps)
            
            # Analyze peer calculation patterns
            peer_patterns = []
            for step_info in communication_steps:
                for peer_expr in step_info['peer_expressions']:
                    if peer_expr and peer_expr not in peer_patterns:
                        peer_patterns.append(peer_expr)
            
            # Detect XOR patterns (recursive doubling indicator)
            xor_patterns = [p for p in peer_patterns if '^' in str(p) or 'xor' in str(p).lower()]
            if xor_patterns:
                pattern_info['loop_type'] = 'recursive_doubling'
                pattern_info['peer_calculation_pattern'] = 'xor'
            else:
                # Check for arithmetic progression (ring pattern)
                if self._is_arithmetic_progression_pattern(peer_patterns):
                    pattern_info['loop_type'] = 'ring'
                    pattern_info['peer_calculation_pattern'] = 'arithmetic'
                else:
                    pattern_info['loop_type'] = 'custom'
                    pattern_info['peer_calculation_pattern'] = 'custom'
        
        return pattern_info
    
    def _get_instruction_value(self, op_metadata: Dict) -> str:
        """Extract instruction value from operation metadata"""
        instruction = op_metadata.get('instruction')
        if hasattr(instruction, 'value'):
            return instruction.value
        return str(instruction)
    
    def _extract_actual_peer_rank(self, op_metadata: Dict) -> int:
        """Extract actual peer rank from operation metadata"""
        try:
            operation = op_metadata.get('operation')
            if not operation:
                return None
            
            # Try to get peer rank from dst or src
            if hasattr(operation, 'dst') and hasattr(operation.dst, 'rank'):
                rank = operation.dst.rank
                if isinstance(rank, int):
                    return rank
            elif hasattr(operation, 'src') and hasattr(operation.src, 'rank'):
                rank = operation.src.rank
                if isinstance(rank, int):
                    return rank
            
            return None
        except Exception:
            return None
    
    def _is_arithmetic_progression_pattern(self, peer_patterns: List[str]) -> bool:
        """Check if peer patterns follow arithmetic progression (ring indicator)"""
        arithmetic_indicators = ['+', '-', '%', 'rank']
        for pattern in peer_patterns:
            pattern_str = str(pattern).lower()
            if any(indicator in pattern_str for indicator in arithmetic_indicators):
                return True
        return False
    
    def _classify_operation_phase(self, op: Op) -> str:
        """Classify DSL operation into algorithm phases"""
        if op.inst == 'copy':
            # Check if it's initialization copy (rank copying own data)
            if (hasattr(op, 'src') and hasattr(op.src, 'buffer') and 
                hasattr(op, 'dst') and hasattr(op.dst, 'buffer')):
                src_buffer = str(op.src.buffer).split('.')[-1] if hasattr(op.src.buffer, 'value') else str(op.src.buffer)
                dst_buffer = str(op.dst.buffer).split('.')[-1] if hasattr(op.dst.buffer, 'value') else str(op.dst.buffer)
                
                if src_buffer == 'input' and dst_buffer == 'output':
                    return 'initialization'
                else:
                    return 'data_movement'
        elif op.inst in ['send', 'recv']:
            return 'communication'
        elif op.inst in ['reduce', 'rrc', 'rrs']:
            return 'computation'
        else:
            return 'unknown'
    
    def _extract_buffer_info(self, chunk_ref) -> Dict[str, Any]:
        """Extract buffer information from chunk reference"""
        if chunk_ref is None:
            return {'type': 'unknown', 'index': 0, 'size': 1}
        
        buffer_type = 'unknown'
        if hasattr(chunk_ref, 'buffer'):
            buffer_type = str(chunk_ref.buffer).split('.')[-1] if hasattr(chunk_ref.buffer, 'value') else str(chunk_ref.buffer)
        
        return {
            'type': buffer_type,
            'index': getattr(chunk_ref, 'index', 0),
            'size': getattr(chunk_ref, 'size', 1),
            'rank': getattr(chunk_ref, 'rank', None)
        }
    
    def _generate_from_dsl_operations_sequence(self, operations: List, indent_level: int) -> List[str]:
        """Generate code from DSL operations sequence with phase-aware processing"""
        lines = []
        
        if not operations:
            return lines
        
        lines.append(self._indent("// DSL operations transpiled with phase analysis", indent_level))
        
        # Group operations by phase first, then by step
        phase_groups = self._group_operations_by_phase(operations)
        
        for phase_name, phase_ops in phase_groups.items():
            lines.append(self._indent(f"// Phase: {phase_name.upper()}", indent_level))
            
            if phase_name == 'initialization':
                lines.extend(self._generate_initialization_phase_code(phase_ops, indent_level))
            elif phase_name == 'communication':
                lines.extend(self._generate_communication_phase_code(phase_ops, indent_level))
            elif phase_name == 'computation':
                lines.extend(self._generate_computation_phase_code(phase_ops, indent_level))
            else:
                lines.extend(self._generate_generic_phase_code(phase_ops, indent_level))
            
            lines.append(self._indent("", indent_level))
        
        return lines
    
    def _group_operations_by_phase(self, operations: List) -> Dict[str, List]:
        """Group operations by their algorithm phase"""
        phase_groups = {}
        
        for op_metadata in operations:
            if isinstance(op_metadata, dict) and 'phase' in op_metadata:
                phase = op_metadata['phase']
            else:
                # Handle direct operation objects
                phase = self._classify_operation_phase(op_metadata)
                
            if phase not in phase_groups:
                phase_groups[phase] = []
            phase_groups[phase].append(op_metadata)
        
        # Ensure phases are ordered logically
        ordered_phases = {}
        for phase in ['initialization', 'communication', 'computation', 'data_movement', 'unknown']:
            if phase in phase_groups:
                ordered_phases[phase] = phase_groups[phase]
        
        return ordered_phases
    
    def _generate_initialization_phase_code(self, ops: List, indent_level: int) -> List[str]:
        """Generate code for initialization phase operations"""
        lines = []
        lines.append(self._indent("// Initialization: Copy own data to output buffer", indent_level))
        
        for op_metadata in ops:
            op = op_metadata['operation'] if isinstance(op_metadata, dict) else op_metadata
            op_code = self._generate_operation_code(op)
            if op_code:
                lines.append(self._indent(op_code, indent_level + 1))
        
        return lines
    
    def _generate_communication_phase_code(self, ops: List, indent_level: int) -> List[str]:
        """Generate code for communication phase operations"""
        lines = []
        lines.append(self._indent("// Communication: Data exchange between ranks", indent_level))
        
        # Group communication operations by step for proper sequencing
        step_groups = {}
        for op_metadata in ops:
            step = op_metadata['step'] if isinstance(op_metadata, dict) else getattr(op_metadata, 'step', 0)
            if step not in step_groups:
                step_groups[step] = []
            step_groups[step].append(op_metadata)
        
        # Generate communication loops or iterations
        for step in sorted(step_groups.keys()):
            step_ops = step_groups[step]
            lines.append(self._indent(f"// Communication step {step}", indent_level))
            
            for op_metadata in step_ops:
                op = op_metadata['operation'] if isinstance(op_metadata, dict) else op_metadata
                op_code = self._generate_operation_code(op)
                if op_code:
                    lines.append(self._indent(op_code, indent_level + 1))
        
        return lines
    
    def _generate_computation_phase_code(self, ops: List, indent_level: int) -> List[str]:
        """Generate code for computation phase operations"""
        lines = []
        lines.append(self._indent("// Computation: Reduce and local computation operations", indent_level))
        
        for op_metadata in ops:
            op = op_metadata['operation'] if isinstance(op_metadata, dict) else op_metadata
            op_code = self._generate_operation_code(op)
            if op_code:
                lines.append(self._indent(op_code, indent_level + 1))
        
        return lines
    
    def _generate_generic_phase_code(self, ops: List, indent_level: int) -> List[str]:
        """Generate code for generic/unknown phase operations"""
        lines = []
        
        for op_metadata in ops:
            op = op_metadata['operation'] if isinstance(op_metadata, dict) else op_metadata
            op_code = self._generate_operation_code(op)
            if op_code:
                lines.append(self._indent(op_code, indent_level + 1))
        
        return lines
    
    def _generate_ring_algorithm(self, program: Program, analysis: Dict[str, Any], indent_level: int) -> List[str]:
        """Generate ring-specific algorithm code with proper HCCL link usage"""
        lines = []
        
        lines.append(self._indent("// Ring AllGather Algorithm Implementation", indent_level))
        lines.append(self._indent("// Dynamic ring topology with peer-based communication", indent_level))
        lines.append(self._indent("", indent_level))
        
        # Ring algorithm: (rankSize - 1) iterations
        lines.append(self._indent("for (u32 step = 0; step < rankSize - 1; step++) {", indent_level))
        indent_level += 1
        
        # Calculate peer rank for ring communication
        lines.append(self._indent("// Ring communication: send to next rank, receive from previous rank", indent_level))
        lines.append(self._indent("u32 sendPeer = (rank + 1) % rankSize;", indent_level))
        lines.append(self._indent("u32 recvPeer = (rank - 1 + rankSize) % rankSize;", indent_level))
        lines.append(self._indent("", indent_level))
        
        # Calculate chunk indices for ring forwarding
        lines.append(self._indent("// Chunk forwarding pattern in ring", indent_level))
        lines.append(self._indent("u32 sendChunkIdx = (rank - step + rankSize) % rankSize;", indent_level))
        lines.append(self._indent("u32 recvChunkIdx = (rank - step - 1 + rankSize) % rankSize;", indent_level))
        lines.append(self._indent("", indent_level))
        
        # Memory management
        lines.append(self._indent("// Prepare memory for send and receive operations", indent_level))
        lines.append(self._indent("u64 chunkSize = sliceSize;", indent_level))
        lines.append(self._indent("DeviceMem srcMem = outputMem_.range(sendChunkIdx * chunkSize, chunkSize);", indent_level))
        lines.append(self._indent("DeviceMem dstMem = outputMem_.range(recvChunkIdx * chunkSize, chunkSize);", indent_level))
        lines.append(self._indent("", indent_level))
        
        # Link selection with bounds checking
        lines.append(self._indent("// Dynamic link selection for ring peers", indent_level))
        lines.append(self._indent("if (sendPeer >= links.size() || recvPeer >= links.size()) {", indent_level))
        lines.append(self._indent(f'    HCCL_ERROR("[{self.config.class_name}][Ring] peer out of bounds: send[%u] recv[%u] linkSize[%zu]",', indent_level + 1))
        lines.append(self._indent("        sendPeer, recvPeer, links.size());", indent_level + 1))
        lines.append(self._indent("    return HCCL_E_INTERNAL;", indent_level + 1))
        lines.append(self._indent("}", indent_level))
        lines.append(self._indent("", indent_level))
        
        # Corrected Ack protocol for ring communication
        lines.append(self._indent("// Corrected Ack protocol: TxAck to receiver, RxAck to sender", indent_level))
        lines.append(self._indent("CHK_RET(links[recvPeer]->TxAck(stream_));  // Tell receiver 'I'm ready to send'", indent_level))
        lines.append(self._indent("CHK_RET(links[sendPeer]->RxAck(stream_));  // Tell sender 'I'm ready to receive'", indent_level))
        lines.append(self._indent("", indent_level))
        
        lines.append(self._indent("CHK_RET(links[sendPeer]->TxAsync(UserMemType::OUTPUT_MEM,", indent_level))
        lines.append(self._indent("    sendChunkIdx * chunkSize + baseOffset_, srcMem.ptr(), chunkSize, stream_));", indent_level))
        lines.append(self._indent("CHK_RET(links[recvPeer]->RxAsync(UserMemType::OUTPUT_MEM,", indent_level))
        lines.append(self._indent("    recvChunkIdx * chunkSize + baseOffset_, dstMem.ptr(), chunkSize, stream_));", indent_level))
        lines.append(self._indent("", indent_level))
        
        # Synchronization - wait for receive first, then send
        lines.append(self._indent("// Wait for communication completion", indent_level))
        lines.append(self._indent("CHK_RET(links[recvPeer]->RxWaitDone(stream_));  // Wait for receive completion first", indent_level))
        lines.append(self._indent("CHK_RET(links[sendPeer]->TxWaitDone(stream_)); // Then wait for send completion", indent_level))
        
        indent_level -= 1
        lines.append(self._indent("}", indent_level))
        
        return lines
    
    def _generate_mesh_algorithm_from_dsl(self, program: Program, analysis: Dict[str, Any], indent_level: int) -> List[str]:
        """Generate mesh algorithm code based on DSL operation analysis"""
        lines = []
        
        lines.append(self._indent("// Mesh AllGather Algorithm Implementation", indent_level))
        lines.append(self._indent("// DSL-driven fully connected topology communication", indent_level))
        lines.append(self._indent("", indent_level))
        
        # Extract and analyze DSL operations
        dsl_operations = self._extract_dsl_operations(program)
        loop_analysis = self._analyze_dsl_loop_structure(program)
        
        # Generate code based on DSL loop structure
        if loop_analysis['has_loops']:
            lines.extend(self._generate_mesh_loop_from_dsl(dsl_operations, loop_analysis, indent_level))
        else:
            # Fallback to step-by-step operation generation
            lines.extend(self._generate_mesh_operations_from_dsl(dsl_operations, indent_level))
        
        return lines
    
    def _generate_alltoall_algorithm_from_dsl(self, program: Program, analysis: Dict[str, Any], indent_level: int) -> List[str]:
        """Generate AllToAll algorithm code based on DSL operation analysis"""
        lines = []
        
        # Generate correct algorithm comment based on detected pattern
        if analysis.get('pattern') == 'recursive_doubling':
            lines.append(self._indent("// Bruck AllToAll Algorithm Implementation", indent_level))
            lines.append(self._indent("// DSL-driven XOR-based parallel data exchange", indent_level))
        else:
            lines.append(self._indent("// AllToAll Algorithm Implementation", indent_level))
            lines.append(self._indent("// DSL-driven all-to-all communication", indent_level))
        lines.append(self._indent("", indent_level))
        
        # Extract and analyze DSL operations
        dsl_operations = self._extract_dsl_operations(program)
        loop_analysis = self._analyze_dsl_loop_structure(program)
        
        # Analyze operations to extract AllToAll communication pattern
        operations_by_step = {}
        for op_metadata in dsl_operations:
            step = op_metadata['step']
            if step not in operations_by_step:
                operations_by_step[step] = []
            operations_by_step[step].append(op_metadata)
        
        # Generate algorithm based on actual DSL semantics
        if analysis.get('pattern') == 'recursive_doubling':
            lines.extend(self._generate_bruck_alltoall_from_dsl(operations_by_step, analysis, indent_level))
        else:
            lines.extend(self._generate_generic_alltoall_from_dsl(operations_by_step, analysis, indent_level))
        
        return lines
    
    def _generate_bruck_alltoall_from_dsl(self, operations_by_step: Dict[int, List], analysis: Dict[str, Any], indent_level: int) -> List[str]:
        """Generate Bruck AllToAll algorithm from DSL operations"""
        lines = []
        
        # Calculate number of steps from DSL operations
        max_step = max(operations_by_step.keys()) if operations_by_step else 0
        num_steps = max_step + 1
        
        lines.append(self._indent("// Bruck algorithm: log2(N) steps with XOR-based communication", indent_level))
        lines.append(self._indent("u32 numSteps = static_cast<u32>(log2(rankSize)); // log2(rankSize)", indent_level))
        lines.append(self._indent("", indent_level))
        
        lines.append(self._indent("for (u32 step = 0; step < numSteps; step++) {", indent_level))
        lines.append(self._indent("u32 distance = 1 << step; // 2^step", indent_level + 1))
        lines.append(self._indent("u32 peerRank = rank ^ distance; // XOR with distance", indent_level + 1))
        lines.append(self._indent("", indent_level + 1))
        
        lines.append(self._indent("if (peerRank < rankSize && peerRank < links.size()) {", indent_level + 1))
        
        # Generate send operations based on DSL semantics
        lines.append(self._indent("// Send chunks based on bit patterns (Bruck algorithm)", indent_level + 2))
        lines.append(self._indent("for (u32 dstRank = 0; dstRank < rankSize; dstRank++) {", indent_level + 2))
        lines.append(self._indent("// Check if bit 'step' of dstRank differs between rank and peerRank", indent_level + 3))
        lines.append(self._indent("if (((dstRank >> step) & 1) != ((rank >> step) & 1)) {", indent_level + 3))
        
        lines.append(self._indent("u64 chunkSize = sliceSize;", indent_level + 4))
        lines.append(self._indent("DeviceMem srcMem = outputMem_.range(dstRank * chunkSize, chunkSize);", indent_level + 4))
        lines.append(self._indent("", indent_level + 4))
        lines.append(self._indent("CHK_RET(links[peerRank]->TxAck(stream_));", indent_level + 4))
        lines.append(self._indent("CHK_RET(links[peerRank]->TxAsync(UserMemType::OUTPUT_MEM,", indent_level + 4))
        lines.append(self._indent("    dstRank * chunkSize + baseOffset_, srcMem.ptr(), chunkSize, stream_));", indent_level + 4))
        lines.append(self._indent("CHK_RET(links[peerRank]->TxWaitDone(stream_));", indent_level + 4))
        
        lines.append(self._indent("}", indent_level + 3))
        lines.append(self._indent("}", indent_level + 2))
        
        # Generate receive operations
        lines.append(self._indent("", indent_level + 2))
        lines.append(self._indent("// Receive chunks from peer", indent_level + 2))
        lines.append(self._indent("for (u32 srcRank = 0; srcRank < rankSize; srcRank++) {", indent_level + 2))
        lines.append(self._indent("if (((srcRank >> step) & 1) != ((peerRank >> step) & 1)) {", indent_level + 3))
        
        lines.append(self._indent("u64 chunkSize = sliceSize;", indent_level + 4))
        lines.append(self._indent("DeviceMem dstMem = outputMem_.range(srcRank * chunkSize, chunkSize);", indent_level + 4))
        lines.append(self._indent("", indent_level + 4))
        lines.append(self._indent("CHK_RET(links[peerRank]->RxAck(stream_));", indent_level + 4))
        lines.append(self._indent("CHK_RET(links[peerRank]->RxAsync(UserMemType::OUTPUT_MEM,", indent_level + 4))
        lines.append(self._indent("    srcRank * chunkSize + baseOffset_, dstMem.ptr(), chunkSize, stream_));", indent_level + 4))
        lines.append(self._indent("CHK_RET(links[peerRank]->RxWaitDone(stream_));", indent_level + 4))
        
        lines.append(self._indent("}", indent_level + 3))
        lines.append(self._indent("}", indent_level + 2))
        
        lines.append(self._indent("}", indent_level + 1))
        lines.append(self._indent("}", indent_level))
        
        return lines
    
    def _generate_generic_alltoall_from_dsl(self, operations_by_step: Dict[int, List], analysis: Dict[str, Any], indent_level: int) -> List[str]:
        """Generate generic AllToAll algorithm from DSL operations"""
        lines = []
        
        lines.append(self._indent("// Generic AllToAll: each rank sends to all other ranks", indent_level))
        lines.append(self._indent("for (u32 dstRank = 0; dstRank < rankSize; dstRank++) {", indent_level))
        lines.append(self._indent("if (dstRank != rank && dstRank < links.size()) {", indent_level + 1))
        
        lines.append(self._indent("u64 chunkSize = sliceSize;", indent_level + 2))
        lines.append(self._indent("DeviceMem srcMem = outputMem_.range(dstRank * chunkSize, chunkSize);", indent_level + 2))
        lines.append(self._indent("", indent_level + 2))
        lines.append(self._indent("CHK_RET(links[dstRank]->TxAck(stream_));", indent_level + 2))
        lines.append(self._indent("CHK_RET(links[dstRank]->TxAsync(UserMemType::OUTPUT_MEM,", indent_level + 2))
        lines.append(self._indent("    dstRank * chunkSize + baseOffset_, srcMem.ptr(), chunkSize, stream_));", indent_level + 2))
        lines.append(self._indent("CHK_RET(links[dstRank]->TxWaitDone(stream_));", indent_level + 2))
        
        lines.append(self._indent("}", indent_level + 1))
        lines.append(self._indent("}", indent_level))
        
        return lines
    
    def _generate_mesh_loop_from_dsl(self, dsl_operations: List, loop_analysis: Dict[str, Any], indent_level: int) -> List[str]:
        """Generate mesh loop structure based on DSL loop analysis"""
        lines = []
        
        # Group operations by step for mesh pattern analysis
        operations_by_step = {}
        for op_metadata in dsl_operations:
            step = op_metadata['step']
            if step not in operations_by_step:
                operations_by_step[step] = []
            operations_by_step[step].append(op_metadata)
        
        max_steps = max(operations_by_step.keys()) if operations_by_step else 0
        
        if max_steps > 0:
            # Generate mesh communication loop based on DSL structure
            lines.append(self._indent(f"for (u32 step = 0; step < {max_steps + 1}; step++) {{", indent_level))
            indent_level += 1
            
            # Analyze mesh communication pattern from DSL operations
            comm_pattern = self._analyze_mesh_communication_pattern_from_dsl(operations_by_step)
            
            # Generate peer calculation based on DSL analysis
            if comm_pattern['peer_calculation'] == 'round_robin':
                lines.append(self._indent("// Mesh round-robin communication pattern from DSL", indent_level))
                lines.append(self._indent("u32 srcRank = (rank + step + 1) % rankSize;", indent_level))
            elif comm_pattern['peer_calculation'] == 'arithmetic':
                lines.append(self._indent("// Mesh arithmetic communication pattern from DSL", indent_level))  
                lines.append(self._indent("u32 srcRank = (rank + step) % rankSize;", indent_level))
            else:
                # Default mesh pattern
                lines.append(self._indent("// Default mesh communication pattern", indent_level))
                lines.append(self._indent("u32 srcRank = (rank + step + 1) % rankSize;", indent_level))
            
            lines.append(self._indent("if (srcRank == rank) continue; // Skip self", indent_level))
            lines.append(self._indent("", indent_level))
            
            # Generate memory operations based on DSL buffer analysis
            lines.extend(self._generate_mesh_memory_operations_from_dsl(operations_by_step, indent_level))
            
            # Generate communication operations
            lines.extend(self._generate_mesh_communication_from_dsl(comm_pattern, indent_level))
            
            indent_level -= 1
            lines.append(self._indent("}", indent_level))
        else:
            lines.append(self._indent("// No step-based operations found in DSL", indent_level))
        
        return lines
    
    def _analyze_mesh_communication_pattern_from_dsl(self, operations_by_step: Dict[int, List]) -> Dict[str, Any]:
        """Analyze mesh communication pattern from DSL operations"""
        pattern = {
            'peer_calculation': 'round_robin',
            'buffer_usage': 'output',
            'memory_pattern': 'chunked',
            'communication_type': 'all_to_all'
        }
        
        # Analyze peer expressions in operations
        peer_expressions = []
        for step, ops in operations_by_step.items():
            for op_metadata in ops:
                if op_metadata['peer_expression']:
                    peer_expressions.append(op_metadata['peer_expression'])
        
        # Determine peer calculation pattern from DSL
        if any('xor' in str(expr).lower() or '^' in str(expr) for expr in peer_expressions):
            pattern['peer_calculation'] = 'xor'
        elif any('+' in str(expr) and 'step' in str(expr) for expr in peer_expressions):
            pattern['peer_calculation'] = 'round_robin'
        elif any('-' in str(expr) or '%' in str(expr) for expr in peer_expressions):
            pattern['peer_calculation'] = 'arithmetic'
        
        return pattern
    
    def _generate_mesh_memory_operations_from_dsl(self, operations_by_step: Dict[int, List], indent_level: int) -> List[str]:
        """Generate memory operations for mesh AllGather with both send and receive buffers"""
        lines = []
        
        lines.append(self._indent("// Memory preparation for AllGather bidirectional communication", indent_level))
        lines.append(self._indent("u64 chunkSize = sliceSize;", indent_level))
        lines.append(self._indent("DeviceMem srcMem = outputMem_.range(rank * chunkSize, chunkSize);    // Own data to send", indent_level))
        lines.append(self._indent("DeviceMem dstMem = outputMem_.range(srcRank * chunkSize, chunkSize); // Buffer for received data", indent_level))
        lines.append(self._indent("", indent_level))
        
        return lines
    
    def _generate_mesh_communication_from_dsl(self, comm_pattern: Dict[str, Any], indent_level: int) -> List[str]:
        """Generate parallel mesh communication operations - launch all, then wait all"""
        lines = []
        
        lines.append(self._indent("// Parallel mesh AllGather communication", indent_level))
        lines.append(self._indent("u64 chunkSize = sliceSize;", indent_level))
        lines.append(self._indent("", indent_level))
        
        # Phase 1: Launch all communications
        lines.append(self._indent("// Phase 1: Launch all peer communications in parallel", indent_level))
        lines.append(self._indent("for (u32 peerRank = 0; peerRank < rankSize; peerRank++) {", indent_level))
        lines.append(self._indent("    if (peerRank == rank || peerRank >= links.size()) continue;", indent_level + 1))
        lines.append(self._indent("", indent_level + 1))
        
        # Memory setup for each peer
        lines.append(self._indent("    // Memory setup for bidirectional communication", indent_level + 1))
        lines.append(self._indent("    DeviceMem srcMem = outputMem_.range(rank * chunkSize, chunkSize);     // Own data to send", indent_level + 1))
        lines.append(self._indent("    DeviceMem dstMem = outputMem_.range(peerRank * chunkSize, chunkSize); // Buffer for peer's data", indent_level + 1))
        lines.append(self._indent("", indent_level + 1))
        
        # Asymmetric handshake protocol to avoid race condition deadlock
        lines.append(self._indent("    // Asymmetric handshake protocol (rank ID-based ordering to avoid deadlock)", indent_level + 1))
        lines.append(self._indent("    if (rank < peerRank) {", indent_level + 1))
        lines.append(self._indent("        // Lower rank ID: initiate handshake first", indent_level + 2))
        lines.append(self._indent("        CHK_RET(links[peerRank]->TxAck(stream_));  // Signal: I'm ready to receive", indent_level + 2))
        lines.append(self._indent("        CHK_RET(links[peerRank]->RxAck(stream_));  // Wait: peer is ready to send", indent_level + 2))
        lines.append(self._indent("    } else {", indent_level + 1))
        lines.append(self._indent("        // Higher rank ID: respond to handshake", indent_level + 2))
        lines.append(self._indent("        CHK_RET(links[peerRank]->RxAck(stream_));  // Wait: peer initiates handshake", indent_level + 2))
        lines.append(self._indent("        CHK_RET(links[peerRank]->TxAck(stream_));  // Response: I'm ready to send", indent_level + 2))
        lines.append(self._indent("    }", indent_level + 1))
        lines.append(self._indent("", indent_level + 1))
        
        lines.append(self._indent("    // Launch async send (own data to peer)", indent_level + 1))
        lines.append(self._indent("    CHK_RET(links[peerRank]->TxAsync(UserMemType::OUTPUT_MEM,", indent_level + 1))
        lines.append(self._indent("        rank * chunkSize + baseOffset_, srcMem.ptr(), chunkSize, stream_));", indent_level + 1))
        lines.append(self._indent("", indent_level + 1))
        
        lines.append(self._indent("    // Launch async receive (peer's data)", indent_level + 1))
        lines.append(self._indent("    CHK_RET(links[peerRank]->RxAsync(UserMemType::OUTPUT_MEM,", indent_level + 1))
        lines.append(self._indent("        peerRank * chunkSize + baseOffset_, dstMem.ptr(), chunkSize, stream_));", indent_level + 1))
        lines.append(self._indent("}", indent_level))
        lines.append(self._indent("", indent_level))
        
        # Phase 2: Wait for all communications to complete
        lines.append(self._indent("// Phase 2: Wait for all communications to complete", indent_level))
        lines.append(self._indent("for (u32 peerRank = 0; peerRank < rankSize; peerRank++) {", indent_level))
        lines.append(self._indent("    if (peerRank == rank || peerRank >= links.size()) continue;", indent_level + 1))
        lines.append(self._indent("", indent_level + 1))
        
        lines.append(self._indent("    // Wait for send completion", indent_level + 1))
        lines.append(self._indent("    CHK_RET(links[peerRank]->TxWaitDone(stream_));", indent_level + 1))
        lines.append(self._indent("", indent_level + 1))
        
        lines.append(self._indent("    // Wait for receive completion", indent_level + 1))
        lines.append(self._indent("    CHK_RET(links[peerRank]->RxWaitDone(stream_));", indent_level + 1))
        lines.append(self._indent("}", indent_level))
        lines.append(self._indent("", indent_level))
        
        return lines
    
    def _generate_mesh_operations_from_dsl(self, dsl_operations: List, indent_level: int) -> List[str]:
        """Generate mesh operations with parallel AllGather communication"""
        lines = []
        
        lines.append(self._indent("// Mesh AllGather with parallel communication pattern", indent_level))
        lines.append(self._indent("// DSL-driven fully connected topology - all ranks communicate with all others", indent_level))
        lines.append(self._indent("", indent_level))
        
        # Extract unique communication relationships from DSL operations
        communication_pairs = set()
        for op_metadata in dsl_operations:
            if isinstance(op_metadata, dict):
                operation = op_metadata['operation']
                gpu_rank = op_metadata['gpu_rank']
                if hasattr(operation, 'dst') and hasattr(operation.dst, 'rank'):
                    dst_rank = operation.dst.rank
                    if dst_rank != gpu_rank:  # Exclude self-communication
                        communication_pairs.add((gpu_rank, dst_rank))
        
        # Generate parallel mesh AllGather algorithm
        if communication_pairs:
            lines.append(self._indent("u64 chunkSize = sliceSize;", indent_level))
            lines.append(self._indent("", indent_level))
            
            # Phase 1: Launch all communications in parallel
            lines.append(self._indent("// Phase 1: Launch all communications in parallel (no blocking)", indent_level))
            lines.append(self._indent("for (u32 peerRank = 0; peerRank < rankSize; peerRank++) {", indent_level))
            lines.append(self._indent("    if (peerRank == rank || peerRank >= links.size()) continue;", indent_level + 1))
            lines.append(self._indent("", indent_level + 1))
            
            # Memory setup for each peer
            lines.append(self._indent("    // Memory setup for bidirectional communication", indent_level + 1))
            lines.append(self._indent("    DeviceMem srcMem = outputMem_.range(rank * chunkSize, chunkSize);     // Own data to send", indent_level + 1))
            lines.append(self._indent("    DeviceMem dstMem = outputMem_.range(peerRank * chunkSize, chunkSize); // Buffer for peer's data", indent_level + 1))
            lines.append(self._indent("", indent_level + 1))
            
            # Asymmetric handshake protocol to avoid race condition deadlock
            lines.append(self._indent("    // Asymmetric handshake protocol (rank ID-based ordering to avoid deadlock)", indent_level + 1))
            lines.append(self._indent("    if (rank < peerRank) {", indent_level + 1))
            lines.append(self._indent("        // Lower rank ID: initiate handshake first", indent_level + 2))
            lines.append(self._indent("        CHK_RET(links[peerRank]->TxAck(stream_));  // Signal: I'm ready to receive", indent_level + 2))
            lines.append(self._indent("        CHK_RET(links[peerRank]->RxAck(stream_));  // Wait: peer is ready to send", indent_level + 2))
            lines.append(self._indent("    } else {", indent_level + 1))
            lines.append(self._indent("        // Higher rank ID: respond to handshake", indent_level + 2))
            lines.append(self._indent("        CHK_RET(links[peerRank]->RxAck(stream_));  // Wait: peer initiates handshake", indent_level + 2))
            lines.append(self._indent("        CHK_RET(links[peerRank]->TxAck(stream_));  // Response: I'm ready to send", indent_level + 2))
            lines.append(self._indent("    }", indent_level + 1))
            lines.append(self._indent("", indent_level + 1))
            
            lines.append(self._indent("    // Launch async send (own data to peer)", indent_level + 1))
            lines.append(self._indent("    CHK_RET(links[peerRank]->TxAsync(UserMemType::OUTPUT_MEM,", indent_level + 1))
            lines.append(self._indent("        rank * chunkSize + baseOffset_, srcMem.ptr(), chunkSize, stream_));", indent_level + 1))
            lines.append(self._indent("", indent_level + 1))
            
            lines.append(self._indent("    // Launch async receive (peer's data)", indent_level + 1))
            lines.append(self._indent("    CHK_RET(links[peerRank]->RxAsync(UserMemType::OUTPUT_MEM,", indent_level + 1))
            lines.append(self._indent("        peerRank * chunkSize + baseOffset_, dstMem.ptr(), chunkSize, stream_));", indent_level + 1))
            lines.append(self._indent("}", indent_level))
            lines.append(self._indent("", indent_level))
            
            # Phase 2: Wait for all communications to complete
            lines.append(self._indent("// Phase 2: Wait for all communications to complete", indent_level))
            lines.append(self._indent("for (u32 peerRank = 0; peerRank < rankSize; peerRank++) {", indent_level))
            lines.append(self._indent("    if (peerRank == rank || peerRank >= links.size()) continue;", indent_level + 1))
            lines.append(self._indent("", indent_level + 1))
            
            lines.append(self._indent("    // Wait for send completion", indent_level + 1))
            lines.append(self._indent("    CHK_RET(links[peerRank]->TxWaitDone(stream_));", indent_level + 1))
            lines.append(self._indent("", indent_level + 1))
            
            lines.append(self._indent("    // Wait for receive completion", indent_level + 1))
            lines.append(self._indent("    CHK_RET(links[peerRank]->RxWaitDone(stream_));", indent_level + 1))
            lines.append(self._indent("}", indent_level))
        else:
            lines.append(self._indent("// No communication pairs found in DSL operations", indent_level))
        
        return lines
    
    def _generate_generic_dsl_algorithm(self, program: Program, indent_level: int) -> List[str]:
        """Generate generic algorithm from DSL operations"""
        lines = []
        
        lines.append(self._indent("// Generic DSL-based Algorithm Implementation", indent_level))
        
        # Collect and process all operations by step
        all_operations = {}
        for gpu in program.gpus:
            for tb in gpu.threadblocks:
                for op in tb.ops:
                    step_id = op.step
                    if step_id not in all_operations:
                        all_operations[step_id] = []
                    all_operations[step_id].append(op)
        
        # Generate code for each step
        for step_id in sorted(all_operations.keys()):
            operations = all_operations[step_id]
            lines.append(self._indent(f"// Step {step_id}", indent_level))
            
            for op in operations:
                operation_code = self._generate_operation_code(op)
                if operation_code:
                    lines.append(self._indent(operation_code, indent_level))
        
        return lines

    def _op_to_cpp(self, op: Op, indent_level: int) -> str:
        """Converts a single hcclang Op to a C++ line of code."""
        lines = []
        try:
            if op.inst == "send":
                src_offset = self._chunk_ref_to_cpp(op.src) if hasattr(op, 'src') and op.src else "slices_[rank].offset"
                lines.append(self._indent(f"// Send operation: rank to rank + 1", indent_level))
                lines.append(self._indent(f"{{", indent_level))
                lines.append(self._indent(f"    Slice txSlice = {{ .offset = {src_offset}, .size = sliceSize }};", indent_level))
                lines.append(self._indent(f"    CHK_RET(Tx(linkRight_, txSlice));", indent_level))
                lines.append(self._indent(f"}}", indent_level))
            elif op.inst == "recv":
                dst_offset = self._chunk_ref_to_cpp(op.dst) if hasattr(op, 'dst') and op.dst else "slices_[rank].offset"
                lines.append(self._indent(f"// Recv operation: rank from rank - 1", indent_level))
                lines.append(self._indent(f"{{", indent_level))
                lines.append(self._indent(f"    Slice rxSlice = {{ .offset = {dst_offset}, .size = sliceSize }};", indent_level))
                lines.append(self._indent(f"    CHK_RET(Rx(linkLeft_, rxSlice));", indent_level))
                lines.append(self._indent(f"}}", indent_level))
            elif op.inst == "copy":
                src_offset = self._chunk_ref_to_cpp(op.src) if hasattr(op, 'src') and op.src else "slices_[rank].offset"
                dst_offset = self._chunk_ref_to_cpp(op.dst) if hasattr(op, 'dst') and op.dst else "slices_[rank].offset"
                lines.append(self._indent(f"// Copy operation", indent_level))
                lines.append(self._indent(f"{{", indent_level))
                lines.append(self._indent(f"    DeviceMem dst = outputMem_.range({dst_offset}, sliceSize);", indent_level))
                lines.append(self._indent(f"    DeviceMem src = inputMem_.range({src_offset}, sliceSize);", indent_level))
                lines.append(self._indent(f"    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, stream_));", indent_level))
                lines.append(self._indent(f"}}", indent_level))
            else:
                lines.append(self._indent(f"// Unsupported op: {op.inst}", indent_level))
        except Exception as e:
            lines.append(self._indent(f"// Error processing op: {e}", indent_level))
        
        return "\n".join(lines)

    def _chunk_ref_to_cpp(self, chunk_ref) -> str:
        """Converts a ChunkRef to a C++ offset expression."""
        if chunk_ref is None:
            return "slices_[rank].offset"
        
        # Handle different types of chunk references
        try:
            if hasattr(chunk_ref, 'rank_str'):
                if chunk_ref.rank_str == "g.rank":
                    return f"slices_[rank].offset"
                # Handle arithmetic in rank_str, e.g., "g.rank - step - 1"
                expr = chunk_ref.rank_str.replace("g.rank", "rank")
                return f"slices_[({expr} + rankSize) % rankSize].offset"
            elif hasattr(chunk_ref, 'rank') and hasattr(chunk_ref, 'buffer'):
                # Handle direct rank reference
                return f"slices_[{chunk_ref.rank}].offset"
            else:
                # Fallback for unknown chunk_ref types
                return f"slices_[rank].offset"
        except Exception:
            return f"slices_[rank].offset"

    def _indent(self, line: str, level: int) -> str:
        """Indents a line of code."""
        return "    " * level + line
    
    def _extract_peer_expression(self, op: Op) -> str:
        """Extract peer calculation expression from DSL operation"""
        # Check destination chunk reference for peer expression
        if hasattr(op, 'dst') and hasattr(op.dst, 'rank_str'):
            return op.dst.rank_str
        elif hasattr(op, 'src') and hasattr(op.src, 'rank_str'):
            return op.src.rank_str
        elif hasattr(op, 'dst') and hasattr(op.dst, 'rank'):
            # Check if rank is a computed value (like XOR result)
            rank_val = op.dst.rank
            if hasattr(rank_val, '__xor__'):  # If it's the result of XOR operation
                return f"rank ^ count"  # This indicates XOR pattern
            return str(rank_val)
        elif hasattr(op, 'src') and hasattr(op.src, 'rank'):
            rank_val = op.src.rank
            if hasattr(rank_val, '__xor__'):
                return f"rank ^ count"
            return str(rank_val)
        
        # Debug print to understand what we're getting
        # print(f"DEBUG: Op {op.inst}, src: {getattr(op, 'src', None)}, dst: {getattr(op, 'dst', None)}")
        
        return ""
    
    def _detect_pattern_from_peer_expressions(self, peer_expressions: List[str], num_ranks: int) -> Dict[str, str]:
        """Detect algorithm pattern by analyzing peer calculation expressions"""
        if not peer_expressions:
            return {
                'pattern': 'generic',
                'topology_type': 'unknown',
                'communication_pattern': 'unknown',
                'peer_calculation': 'unknown'
            }
        
        # Count different types of patterns
        xor_patterns = 0
        arithmetic_patterns = 0
        direct_patterns = 0
        
        for expr in peer_expressions:
            expr_lower = expr.lower()
            
            # Detect XOR patterns (recursive doubling indicator)
            if '^' in expr or 'xor' in expr_lower or '⊕' in expr:
                xor_patterns += 1
            # Detect arithmetic patterns (ring/linear algorithms)
            elif '+' in expr or '-' in expr or '%' in expr:
                arithmetic_patterns += 1
            # Detect direct rank patterns
            elif expr.isdigit():
                direct_patterns += 1
        
        total_patterns = len(peer_expressions)
        
        # Pattern classification based on dominant expression type
        if xor_patterns > total_patterns * 0.6:
            # Majority XOR patterns suggest recursive doubling
            return {
                'pattern': 'recursive_doubling',
                'topology_type': 'fully_connected',
                'communication_pattern': 'all_to_all',
                'peer_calculation': 'xor_distance'
            }
        elif arithmetic_patterns > total_patterns * 0.6:
            # Majority arithmetic patterns suggest ring/linear algorithms
            if self._is_ring_arithmetic_pattern(peer_expressions):
                return {
                    'pattern': 'ring',
                    'topology_type': 'ring',
                    'communication_pattern': 'neighbor',
                    'peer_calculation': 'sequential'
                }
            else:
                return {
                    'pattern': 'linear',
                    'topology_type': 'linear',
                    'communication_pattern': 'sequential',
                    'peer_calculation': 'arithmetic'
                }
        else:
            # Mixed or custom patterns
            return {
                'pattern': 'custom',
                'topology_type': 'custom',
                'communication_pattern': 'mixed',
                'peer_calculation': 'mixed'
            }
    
    def _is_ring_arithmetic_pattern(self, peer_expressions: List[str]) -> bool:
        """Check if arithmetic patterns match ring topology"""
        # Ring patterns typically involve (rank ± 1) % n operations
        ring_indicators = ['+1', '-1', '% rank', 'ranksize']
        
        for expr in peer_expressions:
            expr_lower = expr.lower().replace(' ', '')
            for indicator in ring_indicators:
                if indicator.replace(' ', '') in expr_lower:
                    return True
        return False
    
    def _detect_communication_phases(self, operations_by_step: Dict[int, List]) -> List[Dict[str, Any]]:
        """Detect communication phases from step-wise operations"""
        phases = []
        
        if not operations_by_step:
            return phases
        
        # Analyze operations per step to identify phases
        step_ids = sorted(operations_by_step.keys())
        
        # Phase detection based on operation patterns
        current_phase = None
        current_ops = []
        
        for step_id in step_ids:
            operations = operations_by_step[step_id]
            
            # Classify step operations
            copy_ops = [op for op in operations if op.inst == 'copy']
            send_ops = [op for op in operations if op.inst == 'send']
            recv_ops = [op for op in operations if op.inst == 'recv']
            
            if copy_ops and not (send_ops or recv_ops):
                # Initialization phase - copy operations only
                if current_phase != 'initialization':
                    if current_phase is not None:
                        phases.append({
                            'phase': current_phase,
                            'steps': len(current_ops),
                            'operations': current_ops
                        })
                    current_phase = 'initialization'
                    current_ops = [step_id]
                else:
                    current_ops.append(step_id)
            elif send_ops or recv_ops:
                # Communication phase - send/recv operations
                if current_phase != 'communication':
                    if current_phase is not None:
                        phases.append({
                            'phase': current_phase,
                            'steps': len(current_ops),
                            'operations': current_ops
                        })
                    current_phase = 'communication'
                    current_ops = [step_id]
                else:
                    current_ops.append(step_id)
        
        # Add final phase
        if current_phase is not None:
            phases.append({
                'phase': current_phase,
                'steps': len(current_ops),
                'operations': current_ops
            })
        
        return phases
    
    def _analyze_operation_peer_pattern(self, op: Op) -> str:
        """Analyze operation context to determine peer pattern"""
        # Extract peer information from operation if available
        if hasattr(op, 'peer_rank') and op.peer_rank is not None:
            return f"u32 peer = {op.peer_rank};  // Direct peer from DSL"
        
        # Fallback: check if we can infer from operation structure
        # This should rarely be reached if DSL operations are properly formed
        return "u32 peer = (rank + 1) % rankSize;  // Generic fallback pattern"
    
    def _extract_sendtb_info(self, op: Op) -> str:
        """Extract sendtb parameter information from DSL operation"""
        if hasattr(op, 'sendtb') and op.sendtb is not None:
            return f"\n        // DSL sendtb parameter: {op.sendtb} (threadblock routing)"
        return ""
    
    def _extract_recvtb_info(self, op: Op) -> str:
        """Extract recvtb parameter information from DSL operation"""  
        if hasattr(op, 'recvtb') and op.recvtb is not None:
            return f"\n        // DSL recvtb parameter: {op.recvtb} (threadblock routing)"
        return ""

    
    def _generate_from_dsl_operations(self, program: Program, comm_analysis: Dict[str, Any]) -> str:
        """Generate C++ code directly from DSL operations sequence"""
        if not program.gpus:
            return ""
        
        code_blocks = []
        code_blocks.append("    // Generated from DSL operations sequence")
        
        # Group operations by step
        operations_by_step = comm_analysis['operations_per_step']
        
        if not operations_by_step:
            return ""
        
        # Generate initialization
        code_blocks.append("    u32 sliceSize = outputSlices.size() / rankSize;")
        code_blocks.append("")
        
        # Generate code for each step
        for step_id in sorted(operations_by_step.keys()):
            operations = operations_by_step[step_id]
            code_blocks.append(f"    // Step {step_id}")
            
            # Group operations by type for this step
            step_operations = {
                'send': [],
                'recv': [],
                'copy': [],
                'reduce': [],
                'rrc': [],
                'rrs': []
            }
            
            for op in operations:
                if op.inst in step_operations:
                    step_operations[op.inst].append(op)
            
            # Generate synchronization before data operations
            if step_operations['send'] or step_operations['recv']:
                code_blocks.append("    CHK_RET(linkLeft_->TxAck(stream_));")
                code_blocks.append("    CHK_RET(linkRight_->RxAck(stream_));")
                code_blocks.append("")
            
            # Generate code for each operation type
            for send_op in step_operations['send']:
                op_code = self._generate_dsl_send_operation(send_op, step_id)
                code_blocks.append(op_code)
            
            for recv_op in step_operations['recv']:
                op_code = self._generate_dsl_recv_operation(recv_op, step_id)
                code_blocks.append(op_code)
            
            for copy_op in step_operations['copy']:
                op_code = self._generate_dsl_copy_operation(copy_op, step_id)
                code_blocks.append(op_code)
            
            for reduce_op in step_operations['reduce']:
                op_code = self._generate_dsl_reduce_operation(reduce_op, step_id)
                code_blocks.append(op_code)
            
            for rrc_op in step_operations['rrc']:
                op_code = self._generate_dsl_rrc_operation(rrc_op, step_id)
                code_blocks.append(op_code)
            
            for rrs_op in step_operations['rrs']:
                op_code = self._generate_dsl_rrs_operation(rrs_op, step_id)
                code_blocks.append(op_code)
            
            # Generate synchronization after data operations
            if step_operations['send'] or step_operations['recv']:
                code_blocks.append("")
                code_blocks.append("    CHK_RET(linkLeft_->RxWaitDone(stream_));")
                code_blocks.append("    CHK_RET(linkRight_->TxWaitDone(stream_));")
            
            code_blocks.append("")
        
        return '\n'.join(code_blocks)
    
    def _generate_dsl_send_operation(self, op: Op, step_id: int) -> str:
        """Generate HCCL send operation from DSL send"""
        # Extract buffer and offset information from ChunkRef
        if hasattr(op, 'dst') and op.dst:
            chunk_ref = op.dst
            buffer_name = chunk_ref.buffer.name if hasattr(chunk_ref.buffer, 'name') else 'outputMem'
            offset_expr = f"step_{step_id}_offset" 
        else:
            buffer_name = 'outputMem'
            offset_expr = f"txSliceIndex * sliceSize"
        
        return f"""    // DSL send operation (step {step_id})
    std::vector<Slice> txSegsSlice;
    for (u32 j = 0; j < sliceSize; j++) {{
        if ({offset_expr} + j < outputSlices.size()) {{
            txSegsSlice.push_back(outputSlices[{offset_expr} + j]);
        }}
    }}
    HcclResult ret = TxVector(linkRight_, txSegsSlice);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[{self.config.class_name}]step[{step_id}] send operation failed"), ret);"""
    
    def _generate_dsl_recv_operation(self, op: Op, step_id: int) -> str:
        """Generate HCCL receive operation from DSL recv"""
        # Extract buffer and offset information from ChunkRef
        if hasattr(op, 'dst') and op.dst:
            chunk_ref = op.dst
            buffer_name = chunk_ref.buffer.name if hasattr(chunk_ref.buffer, 'name') else 'outputMem'
            offset_expr = f"step_{step_id}_offset"
        else:
            buffer_name = 'outputMem'
            offset_expr = f"rxSliceIndex * sliceSize"
        
        return f"""    // DSL recv operation (step {step_id})
    std::vector<Slice> rxSegsSlice;
    for (u32 j = 0; j < sliceSize; j++) {{
        if ({offset_expr} + j < outputSlices.size()) {{
            rxSegsSlice.push_back(outputSlices[{offset_expr} + j]);
        }}
    }}
    HcclResult ret = RxVector(linkLeft_, rxSegsSlice);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[{self.config.class_name}]step[{step_id}] recv operation failed"), ret);"""
    
    def _generate_dsl_copy_operation(self, op: Op, step_id: int) -> str:
        """Generate HCCL copy operation from DSL copy"""
        return f"""    // DSL copy operation (step {step_id})
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, stream_));"""
    
    def _generate_dsl_reduce_operation(self, op: Op, step_id: int) -> str:
        """Generate HCCL reduce operation from DSL reduce"""
        return f"""    // DSL reduce operation (step {step_id})
    CHK_RET(linkLeft_->RxAsync(UserMemType::INPUT_MEM, offset, dstMem.ptr(), dataSize, stream_));
    // Local reduction will be handled by the HCCL runtime"""
    
    def _generate_dsl_rrc_operation(self, op: Op, step_id: int) -> str:
        """Generate HCCL receive-reduce-copy operation from DSL rrc"""
        return f"""    // DSL receive-reduce-copy operation (step {step_id})
    CHK_RET(linkLeft_->RxWithReduce(UserMemType::INPUT_MEM, offset, dstMem.ptr(), dataSize, 
        reduceSrc, reduceDst, count, dataType, reduceOp, stream_, attr));"""
    
    def _generate_dsl_rrs_operation(self, op: Op, step_id: int) -> str:
        """Generate HCCL receive-reduce-send operation from DSL rrs"""
        return f"""    // DSL receive-reduce-reduce-send operation (step {step_id})
    CHK_RET(linkLeft_->RxWithReduce(UserMemType::INPUT_MEM, offset, dstMem.ptr(), dataSize, 
        reduceSrc, reduceDst, count, dataType, reduceOp, stream_, attr));
    CHK_RET(linkRight_->TxAsync(UserMemType::OUTPUT_MEM, offset, srcMem.ptr(), dataSize, stream_));"""
    
    # Removed hardcoded _generate_ring_allgather_steps - use DSL transpilation instead

    # Removed hardcoded _generate_multi_ring_allgather_steps - use DSL transpilation instead

    # Removed hardcoded _generate_hierarchical_ring_allgather_steps - use DSL transpilation instead
    
    # def _generate_generic_algorithm_steps(self, program: Program, comm_analysis: Dict[str, Any]) -> str:
    #     """Generate generic algorithm steps from DSL operations"""
    #     # UNUSED METHOD - commented out during refactoring
    #     # This method was not called anywhere in the codebase
    #     steps_code = []
    #     
    #     for step_id in sorted(comm_analysis['operations_per_step'].keys()):
    #         operations = comm_analysis['operations_per_step'][step_id]
    #         steps_code.append(f"    // Step {step_id}")
    #         
    #         for op in operations:
    #             step_code = self._generate_operation_code(op)
    #             if step_code:
    #                 steps_code.append(f"    {step_code}")
    #     
    #     return '\n'.join(steps_code)
    
    def _generate_operation_code(self, op: Op) -> str:
        """Generate C++ code for a single DSL operation"""
        # Map DSL operations to HCCL API calls based on DSL_HCCL_MAPPING.md
        from ..language.ir import Instruction
        
        operation_map = {
            Instruction.send: self._generate_send_operation,
            Instruction.recv: self._generate_recv_operation,
            Instruction.copy: self._generate_copy_operation,
            Instruction.reduce: self._generate_reduce_operation,
            Instruction.recv_reduce_copy: self._generate_rrc_operation,
            Instruction.recv_reduce_send: self._generate_rrs_operation,
            Instruction.recv_copy_send: self._generate_recv_copy_send_operation,
            Instruction.recv_reduce_copy_send: self._generate_recv_reduce_copy_send_operation,
        }
        
        # Add string-based mappings for backward compatibility and missing operations
        string_operation_map = {
            'send_sync': self._generate_send_sync_operation,
            'recv_sync': self._generate_recv_sync_operation,
            'barrier': self._generate_barrier_operation,
            'prepare_tx': self._generate_prepare_tx_operation,
            'prepare_rx': self._generate_prepare_rx_operation,
            'tx_done': self._generate_tx_done_operation,
            'rx_done': self._generate_rx_done_operation,
        }
        
        # Try enum-based mapping first
        generator = operation_map.get(op.inst)
        if generator:
            return generator(op)
        
        # Try string-based mapping for missing operations
        inst_str = str(op.inst) if hasattr(op.inst, 'value') else str(op.inst)
        generator = string_operation_map.get(inst_str)
        if generator:
            return generator(op)
        
        return f"    // TODO: Implement operation {op.inst} - Please check DSL_HCCL_MAPPING.md for correct mapping"
    
    def _generate_send_operation(self, op: Op) -> str:
        """Generate HCCL send operation code with complete synchronization protocol"""
        # Extract source chunk reference and calculate proper slice offset/size
        src_ref = self._chunk_ref_to_cpp(op.src) if hasattr(op, 'src') else "slices_[rank].offset"
        
        # For send operations, determine peer and use appropriate link
        peer_calculation = self._get_peer_calculation(op)
        link_selection = self._get_link_selection(op, peer_calculation)
        
        # Extract DSL parameters for threadblock information
        sendtb_info = self._extract_sendtb_info(op)
        
        return f"""    // DSL send operation with full synchronization protocol
    {{
        u64 srcOffset = {src_ref};
        u64 dataSize = slices_[rank].size;  // Data block size based on DSL semantics
        DeviceMem srcMem = outputMem_.range(srcOffset, dataSize);
        
        {link_selection}
        
        // Step 1: Handshake protocol - ensure receiver is ready
        CHK_RET(peerLink->TxAck(stream_));
        
        // Step 2: Data transfer - async send as per DSL_HCCL_MAPPING.md
        CHK_RET(peerLink->TxAsync(UserMemType::OUTPUT_MEM, srcOffset + baseOffset_, srcMem.ptr(), dataSize, stream_));
        
        // Step 3: Completion synchronization - wait for send completion
        CHK_RET(peerLink->TxWaitDone(stream_));{sendtb_info}
    }}"""
    
    def _generate_recv_operation(self, op: Op) -> str:
        """Generate HCCL receive operation code with complete synchronization protocol"""
        # Extract destination chunk reference
        dst_ref = self._chunk_ref_to_cpp(op.dst) if hasattr(op, 'dst') else "slices_[rank].offset"
        
        # For recv operations, determine peer and use appropriate link
        peer_calculation = self._get_peer_calculation(op)
        link_selection = self._get_link_selection(op, peer_calculation)
        
        # Extract DSL parameters for threadblock information
        recvtb_info = self._extract_recvtb_info(op)
        
        return f"""    // DSL recv operation with full synchronization protocol
    {{
        u64 dstOffset = {dst_ref};
        u64 dataSize = slices_[rank].size;  // Data block size based on DSL semantics
        DeviceMem dstMem = outputMem_.range(dstOffset, dataSize);
        
        {link_selection}
        
        // Step 1: Handshake protocol - signal readiness to receive
        CHK_RET(peerLink->RxAck(stream_));
        
        // Step 2: Data transfer - async receive as per DSL_HCCL_MAPPING.md
        CHK_RET(peerLink->RxAsync(UserMemType::OUTPUT_MEM, dstOffset + baseOffset_, dstMem.ptr(), dataSize, stream_));
        
        // Step 3: Completion synchronization - wait for receive completion
        CHK_RET(peerLink->RxWaitDone(stream_));{recvtb_info}
    }}"""
    
    def _generate_copy_operation(self, op: Op) -> str:
        """Generate HCCL copy operation code based on DSL_HCCL_MAPPING.md"""
        # Extract source and destination chunk information
        src_buffer = str(op.src.buffer).split('.')[-1] if hasattr(op.src.buffer, 'value') else str(op.src.buffer)
        dst_buffer = str(op.dst.buffer).split('.')[-1] if hasattr(op.dst.buffer, 'value') else str(op.dst.buffer)
        
        # Generate appropriate memory references based on buffer types
        if src_buffer == 'input':
            src_mem = f"inputMem_.range({op.src.index} * unitSize, {op.src.size} * unitSize)"
        elif src_buffer == 'output':
            src_mem = f"outputMem_.range({op.src.index} * unitSize, {op.src.size} * unitSize)"
        else:
            src_mem = f"scratchMem_.range({op.src.index} * unitSize, {op.src.size} * unitSize)"
            
        if dst_buffer == 'input':
            dst_mem = f"inputMem_.range({op.dst.index} * unitSize, {op.dst.size} * unitSize)"
        elif dst_buffer == 'output':
            dst_mem = f"outputMem_.range({op.dst.index} * unitSize, {op.dst.size} * unitSize)"
        else:
            dst_mem = f"scratchMem_.range({op.dst.index} * unitSize, {op.dst.size} * unitSize)"
        
        return f"""    // Copy operation: {src_buffer}[{op.src.index}] -> {dst_buffer}[{op.dst.index}] (size: {op.src.size})
    DeviceMem srcMem = {src_mem};
    DeviceMem dstMem = {dst_mem};
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem.ptr(), srcMem.ptr(), srcMem.size(), stream_));"""
    
    def _generate_reduce_operation(self, op: Op) -> str:
        """Generate HCCL reduce operation code"""
        return f"""
// Reduce operation
CHK_RET(linkLeft_->RxAsync(UserMemType::INPUT_MEM, offset, dstMem.ptr(), dataSize, stream_));
// Local reduction will be handled by the HCCL runtime"""
    
    def _generate_rrc_operation(self, op: Op) -> str:
        """Generate HCCL receive-reduce-copy operation code"""
        return f"""
// Receive-reduce-copy operation
CHK_RET(linkLeft_->RxWithReduce(UserMemType::INPUT_MEM, offset, dstMem.ptr(), dataSize, 
    reduceSrc, reduceDst, count, dataType, reduceOp, stream_, attr));"""
    
    def _generate_rrs_operation(self, op: Op) -> str:
        """Generate HCCL receive-reduce-send operation code"""
        return f"""
        // Receive-reduce-send operation
        CHK_RET(linkLeft_->RxWithReduce(UserMemType::INPUT_MEM, offset, dstMem.ptr(), dataSize, 
            reduceSrc, reduceDst, count, dataType, reduceOp, stream_, attr));
        CHK_RET(linkRight_->TxAsync(UserMemType::OUTPUT_MEM, offset, srcMem.ptr(), dataSize, stream_));"""
    
    def _generate_send_sync_operation(self, op: Op) -> str:
        """Generate HCCL synchronous send operation code"""
        return f"""
        // Synchronous send operation
        DeviceMem srcMem = outputMem_.range(srcOffset, dataSize);
        CHK_RET(linkRight_->TxData(UserMemType::OUTPUT_MEM, srcOffset, srcMem.ptr(), dataSize, stream_));"""
    
    def _generate_recv_sync_operation(self, op: Op) -> str:
        """Generate HCCL synchronous receive operation code"""
        return f"""
        // Synchronous receive operation
        DeviceMem dstMem = outputMem_.range(dstOffset, dataSize);
        CHK_RET(linkLeft_->RxData(UserMemType::INPUT_MEM, dstOffset, dstMem.ptr(), dataSize, stream_));"""
    
    def _generate_barrier_operation(self, op: Op) -> str:
        """Generate HCCL barrier operation code"""
        return f"""
        // Barrier operation
        CHK_RET(ExecuteBarrier(linkLeft_, linkRight_));"""
    
    def _generate_prepare_tx_operation(self, op: Op) -> str:
        """Generate HCCL prepare transmission operation code"""
        return f"""
        // Prepare transmission
        CHK_RET(linkRight_->TxPrepare(stream_));"""
    
    def _generate_prepare_rx_operation(self, op: Op) -> str:
        """Generate HCCL prepare reception operation code"""
        return f"""
        // Prepare reception
        CHK_RET(linkLeft_->RxPrepare(stream_));"""
    
    def _generate_tx_done_operation(self, op: Op) -> str:
        """Generate HCCL transmission done operation code"""
        return f"""
        // Signal transmission done
        CHK_RET(linkRight_->TxDone(stream_));"""
    
    def _generate_rx_done_operation(self, op: Op) -> str:
        """Generate HCCL reception done operation code"""
        return f"""
        // Signal reception done
        CHK_RET(linkLeft_->RxDone(stream_));"""
    
    def _generate_recv_copy_send_operation(self, op: Op) -> str:
        """Generate HCCL receive-copy-send operation code"""
        return f"""
        // Receive-copy-send operation
        DeviceMem recvMem = outputMem_.range(recvOffset, dataSize);
        CHK_RET(linkLeft_->RxAsync(UserMemType::INPUT_MEM, recvOffset, recvMem.ptr(), dataSize, stream_));
        CHK_RET(linkLeft_->RxWaitDone(stream_));
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, recvMem, stream_));
        DeviceMem sendMem = outputMem_.range(sendOffset, dataSize);
        CHK_RET(linkRight_->TxAsync(UserMemType::OUTPUT_MEM, sendOffset, sendMem.ptr(), dataSize, stream_));
        CHK_RET(linkRight_->TxWaitDone(stream_));"""
    
    def _generate_recv_reduce_copy_send_operation(self, op: Op) -> str:
        """Generate HCCL receive-reduce-copy-send operation code"""
        return f"""
        // Receive-reduce-copy-send operation
        CHK_RET(linkLeft_->RxWithReduce(UserMemType::INPUT_MEM, recvOffset, recvMem.ptr(), dataSize, 
            reduceSrc, reduceDst, count, dataType, reduceOp, stream_, attr));
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, reduceDst, stream_));
        DeviceMem sendMem = outputMem_.range(sendOffset, dataSize);
        CHK_RET(linkRight_->TxAsync(UserMemType::OUTPUT_MEM, sendOffset, sendMem.ptr(), dataSize, stream_));
        CHK_RET(linkRight_->TxWaitDone(stream_));"""
    
    def _generate_executor_orchestration(self, comm_analysis: Dict[str, Any]) -> str:
        """Generate executor orchestration code (resource management, template instantiation)"""
        lines = []
        
        # Resource management - get data type size
        lines.append("    u32 perDataSize = 0;")
        lines.append("    CHK_RET(SalGetDataTypeSize(param.DataDes.dataType, perDataSize));")
        lines.append("    CHK_PRT_RET(perDataSize == 0,")
        lines.append(f"        HCCL_ERROR(\"[{self.config.class_name}][KernelRun]errNo[0x%016llx] datatype[%d] is invalid\",")
        lines.append("            HCCL_ERROR_CODE(HCCL_E_PARA), param.DataDes.dataType), HCCL_E_PARA);")
        lines.append("")
        
        # Get communication info for level0
        lines.append("    // Get communication info for level0 (intra-server)")
        lines.append("    CHK_RET(CheckCommSize(COMM_LEVEL0, 1));")
        lines.append("    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);")
        lines.append("    u32 localRank = level0CommInfo.localRank;")
        lines.append("    u32 localRankSize = level0CommInfo.localRankSize;")
        lines.append("")
        
        # Remove duplicate data copy - Template already handles this
        lines.append("    // Template will handle input-to-output data copy to avoid duplication")
        lines.append("    u64 inputMemSize = execMem.inputMem.size();")
        
        # Prepare slice information for algorithm
        lines.append("    // Prepare slice information for algorithm")
        lines.append("    std::vector<Slice> dataSlices;")
        lines.append("    u64 sliceSize = inputMemSize;")
        lines.append("    for (u32 i = 0; i < localRankSize; i++) {")
        lines.append("        Slice slice;")
        lines.append("        slice.offset = i * sliceSize;")
        lines.append("        slice.size = sliceSize;")
        lines.append("        dataSlices.push_back(slice);")
        lines.append("    }")
        lines.append("")
        
        # Algorithm template determination - use correct template type that matches registration
        template_type = f"TEMPLATE_{self.config.collective_name_upper}_{self.config.topo_name_upper}"
        lines.append("    // Create and run the algorithm template")
        lines.append("    std::unique_ptr<AlgTemplateBase> algorithmTemplate = ")
        lines.append("        AlgTemplateRegistry::Instance().GetAlgTemplate(")
        lines.append(f"            TemplateType::{template_type}, dispatcher_);")
        lines.append("    CHK_SMART_PTR_NULL(algorithmTemplate);")
        lines.append("")
        
        # Prepare algorithm with parameters
        lines.append("    // Prepare algorithm with parameters")
        lines.append("    CHK_RET(algorithmTemplate->Prepare(execMem.outputMem, execMem.outputMem, execMem.inputMem, ")
        lines.append("        execMem.count, param.DataDes.dataType, param.stream, ")
        lines.append("        HCCL_REDUCE_RESERVED, INVALID_VALUE_RANKID, dataSlices, 0));")
        lines.append("")
        
        # Register profiler
        lines.append("    // Register profiler")
        lines.append("    CHK_RET(algorithmTemplate->RegisterProfiler(")
        lines.append("        (localRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + localRank,")
        lines.append("        PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, param.stream));")
        lines.append("")
        
        # Execute the algorithm
        lines.append("    // Execute the algorithm")
        lines.append("    CHK_RET(RunTemplate(algorithmTemplate, level0CommInfo));")
        lines.append("")
        lines.append(f"    HCCL_INFO(\"[{self.config.class_name}][KernelRun] Algorithm execution completed.\");")
        
        return "\n".join(lines)
    
    def _get_algorithm_template_type(self, comm_analysis: Dict[str, Any]) -> str:
        """Determine the template type based on algorithm analysis"""
        pattern = comm_analysis.get('pattern', 'custom')
        
        if pattern == 'recursive_doubling':
            return 'TEMPLATE_ALLGATHER_RECURSIVE_DOUBLING'
        elif pattern == 'ring':
            return 'TEMPLATE_ALL_GATHER_RING'
        else:
            return 'TEMPLATE_ALLGATHER_CUSTOM'

    def _prepare_template_variables(self, comm_analysis: Dict[str, Any], algorithm_steps: str, executor_orchestration: str) -> Dict[str, Any]:
        """Prepare variables for Jinja2 templates"""
        
        # Calculate required streams based on algorithm structure
        required_streams = 0
        if comm_analysis['is_multi_ring'] and comm_analysis['num_rings'] > 1:
            # Multi-ring algorithms need (num_rings - 1) additional streams
            required_streams = comm_analysis['num_rings'] - 1
        elif comm_analysis['is_hierarchical']:
            # Hierarchical algorithms may need additional streams
            required_streams = 1
        
        # Determine algorithm-specific class names based on detected pattern
        pattern = comm_analysis.get('pattern', 'generic')
        communication_pattern = comm_analysis.get('communication_pattern', self.config.topo_name)
        peer_calculation = comm_analysis.get('peer_calculation', 'sequential')
        
        # Debug: Print pattern detection values to diagnose the problem
        print(f"  DEBUG _prepare_template_variables:")
        print(f"    - pattern: {pattern}")
        print(f"    - communication_pattern: {communication_pattern}")
        print(f"    - peer_calculation: {peer_calculation}")
        
        if pattern == 'recursive_doubling' or (communication_pattern == 'all_to_all' and peer_calculation == 'xor_distance'):
            algorithm_suffix = 'RecursiveDoubling'
            algorithm_type_lower = 'recursive_doubling'
            print(f"    - Selected: RecursiveDoubling algorithm")
        elif pattern == 'ring' or communication_pattern == 'neighbor':
            algorithm_suffix = 'Ring'
            algorithm_type_lower = 'ring'
            print(f"    - Selected: Ring algorithm")
        elif pattern == 'mesh' or communication_pattern == 'all_to_all':
            algorithm_suffix = 'Mesh'
            algorithm_type_lower = 'mesh'
            print(f"    - Selected: Mesh algorithm")
        else:
            algorithm_suffix = self.config.topo_name_camel_case
            algorithm_type_lower = self.config.topo_name
            print(f"    - Selected: Default algorithm ({algorithm_suffix})")
        
        # Override class name and related identifiers
        dynamic_class_name = f"{self.config.collective_name_camel_case}{algorithm_suffix}"
        dynamic_guard_name = f"{self.config.collective_name_upper}_{algorithm_type_lower.upper()}_H"
        dynamic_executor_header_file = f"coll_{self.config.collective_name_lower}_{algorithm_type_lower}_executor.h"
        
        print(f"    - Final class name: {dynamic_class_name}")
        print(f"    - Final algorithm_type_lower: {algorithm_type_lower}")
        
        return {
            # Configuration variables (dynamically overridden based on detected pattern)
            'class_name': dynamic_class_name,
            'guard_name': dynamic_guard_name,
            'collective_name_camel_case': self.config.collective_name_camel_case,
            'collective_name_upper': self.config.collective_name_upper,
            'collective_name_lower': self.config.collective_name_lower,
            'topo_name': algorithm_type_lower,
            'topo_name_upper': algorithm_type_lower.upper(),
            'topo_name_camel_case': algorithm_suffix,
            'executor_header_file': dynamic_executor_header_file,
            'collective_base_name': self.config.collective_base_name,
            'comm_tag': self.config.comm_tag,
            
            # Algorithm-specific variables (for algorithm files)
            'algorithm_steps_code': algorithm_steps,
            'transpiled_algorithm_code': algorithm_steps,  # For algorithm .cc file
            'total_steps': comm_analysis['total_steps'],
            'num_ranks': self.config.num_ranks,
            'algorithm_name': self.config.algorithm_name,
            
            # Executor-specific variables (for executor files)
            'executor_constructor_code': self._generate_executor_constructor_code(),
            'executor_calc_stream_num_code': self._generate_executor_calc_stream_num_code(comm_analysis),
            'executor_calc_comm_info_code': self._generate_executor_calc_comm_info_code(comm_analysis),
            'executor_calc_level0_comm_info_code': self._generate_executor_calc_level0_comm_info_code(comm_analysis),
            'executor_calc_level1_comm_info_code': self._generate_executor_calc_level1_comm_info_code(comm_analysis),
            'executor_calc_level2_comm_info_code': self._generate_executor_calc_level2_comm_info_code(comm_analysis),
            'executor_parse_param_code': self._generate_executor_parse_param_code(),
            'executor_calc_transport_mem_type_code': self._generate_executor_calc_transport_mem_type_code(),
            'executor_calc_loop_max_count_code': self._generate_executor_calc_loop_max_count_code(comm_analysis),
            'executor_is_data_split_code': self._generate_executor_is_data_split_code(),
            'executor_kernel_run_code': executor_orchestration,  # For executor KernelRun method
            'executor_kernel_run_inter_server_code': self._generate_executor_kernel_run_inter_server_code(comm_analysis),
            'executor_kernel_run_intra_server_code': self._generate_executor_kernel_run_intra_server_code(comm_analysis),
            'executor_orchestrate_code': self._generate_executor_orchestrate_code(comm_analysis),
            'executor_additional_interfaces': self._generate_executor_additional_interfaces(comm_analysis),
            'executor_register_name': f"{self.config.collective_name_camel_case}{self.config.topo_name_camel_case}Executor",
            'executor_register_type': f"{self.config.collective_name_camel_case}{self.config.topo_name_camel_case}",
            
            # DSL-generated function placeholders
            'transpiled_algorithm_function': self._generate_dsl_algorithm_function(comm_analysis, algorithm_steps),
            'transpiled_algorithm_function_declarations': self._generate_dsl_algorithm_function_declarations(comm_analysis),
            
            # Enhanced algorithm structure variables
            'num_rings': comm_analysis['num_rings'],
            'is_multi_ring': comm_analysis['is_multi_ring'],
            'is_hierarchical': comm_analysis['is_hierarchical'],
            'ranks_per_ring': comm_analysis['ranks_per_ring'],
            'nodes_per_level': comm_analysis['nodes_per_level'],
            'required_streams': required_streams,
            'communication_phases': comm_analysis['communication_phases'],
            'ring_structure': comm_analysis['ring_structure'],
            
            # Communication pattern for algorithm-specific link setup
            'communication_pattern': comm_analysis.get('communication_pattern', 'mixed'),
            'topology_type': comm_analysis.get('topology_type', 'custom'),
            'peer_calculation': comm_analysis.get('peer_calculation', 'sequential'),
        }
    
    # Removed duplicate methods with algorithm name-based hardcoded logic
    # These methods violated the principle of DSL semantic-based analysis
    
    # Removed _algorithm_name_to_function_name method - violated DSL semantic analysis principle
    
    def _indent_code(self, code: str, indent_level: int) -> str:
        """Indent code lines by specified number of spaces"""
        indent = ' ' * indent_level
        lines = code.split('\n')
        indented_lines = [indent + line if line.strip() else line for line in lines]
        return '\n'.join(indented_lines)
    
    def _generate_algorithm_header(self, template_vars: Dict[str, Any]) -> str:
        """Generate algorithm header file"""
        template = self.jinja_env.get_template('alg.h.j2')
        content = template.render(**template_vars)
        
        # Use detected algorithm pattern for filename instead of hardcoded topology
        pattern = template_vars.get('communication_pattern', self.config.topo_name)
        if pattern == 'all_to_all' and template_vars.get('peer_calculation') == 'xor_distance':
            algorithm_suffix = 'recursive_doubling'
        elif pattern == 'neighbor':
            algorithm_suffix = 'ring'
        elif pattern == 'all_to_all':
            algorithm_suffix = 'mesh'
        else:
            algorithm_suffix = self.config.topo_name
        
        output_file = Path(self.config.output_dir) / f"{self.config.collective_name_lower}_{algorithm_suffix}.h"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            f.write(content)
        
        return str(output_file)
    
    def _generate_algorithm_source(self, template_vars: Dict[str, Any]) -> str:
        """Generate algorithm source file"""
        template = self.jinja_env.get_template('alg.cc.j2')
        content = template.render(**template_vars)
        
        # Use detected algorithm pattern for filename instead of hardcoded topology
        pattern = template_vars.get('communication_pattern', self.config.topo_name)
        if pattern == 'all_to_all' and template_vars.get('peer_calculation') == 'xor_distance':
            algorithm_suffix = 'recursive_doubling'
        elif pattern == 'neighbor':
            algorithm_suffix = 'ring'
        elif pattern == 'all_to_all':
            algorithm_suffix = 'mesh'
        else:
            algorithm_suffix = self.config.topo_name
        
        output_file = Path(self.config.output_dir) / f"{self.config.collective_name_lower}_{algorithm_suffix}.cc"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            f.write(content)
        
        return str(output_file)
    
    def _generate_executor_header(self, template_vars: Dict[str, Any]) -> str:
        """Generate executor header file"""
        template = self.jinja_env.get_template('executor.h.j2')
        content = template.render(**template_vars)
        
        output_file = Path(self.config.output_dir) / f"coll_{self.config.collective_name_lower}_{self.config.topo_name}_executor.h"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            f.write(content)
        
        return str(output_file)
    
    def _generate_executor_source(self, template_vars: Dict[str, Any]) -> str:
        """Generate executor source file"""
        template = self.jinja_env.get_template('executor.cc.j2')
        content = template.render(**template_vars)
        
        output_file = Path(self.config.output_dir) / f"coll_{self.config.collective_name_lower}_{self.config.topo_name}_executor.cc"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            f.write(content)
        
        return str(output_file)


def transpile_dsl_to_hccl(dsl_program: Program, output_dir: str, **kwargs) -> Dict[str, str]:
    """
    Main entry point for transpiling DSL program to HCCL C++ code
    
    Args:
        dsl_program: The DSL program to transpile
        output_dir: Directory to output generated files
        **kwargs: Additional configuration options
        
    Returns:
        Dictionary mapping file types to generated file paths
    """
    # Infer collective and topology from program
    collective = CollectiveType.ALLGATHER  # Default, should be inferred
    topology = TopologyType.RING  # Default, should be inferred
    
    # Get current directory for templates
    current_dir = Path(__file__).parent
    template_dir = current_dir / "templates"
    
    # Create configuration
    config = HcclCodeGenConfig(
        collective=collective,
        topology=topology,
        output_dir=output_dir,
        template_dir=str(template_dir),
        algorithm_name=kwargs.get('algorithm_name', 'CustomAlgorithm'),
        num_ranks=kwargs.get('num_ranks', 4),
        num_steps=kwargs.get('num_steps', 3)
    )
    
    # Create transpiler and generate code
    transpiler = DSLToHcclTranspiler(config)
    return transpiler.transpile_program(dsl_program)


def main():
    """Command-line interface for the HCCLize transpiler"""
    parser = argparse.ArgumentParser(
        description="HCCLize - Transpile DSL algorithms to HCCL C++ code"
    )
    parser.add_argument("input_file", help="Input DSL program file")
    parser.add_argument("-o", "--output", default="./generated", help="Output directory")
    parser.add_argument("--collective", choices=[c.value for c in CollectiveType], 
                       default="allgather", help="Collective operation type")
    parser.add_argument("--topology", choices=[t.value for t in TopologyType], 
                       default="ring", help="Network topology")
    parser.add_argument("--algorithm-name", default="CustomAlgorithm", 
                       help="Algorithm name for generated code")
    parser.add_argument("--num-ranks", type=int, default=4, help="Number of ranks")
    parser.add_argument("--num-steps", type=int, default=3, help="Number of algorithm steps")
    
    args = parser.parse_args()
    
    try:
        # TODO: Implement DSL program parsing from file
        # For now, create a dummy program
        dummy_program = Program(
            gpus=[],
            buffers={},
            name=args.algorithm_name
        )
        
        # Configure transpiler
        config = HcclCodeGenConfig(
            collective=CollectiveType(args.collective),
            topology=TopologyType(args.topology),
            output_dir=args.output,
            template_dir=str(Path(__file__).parent / "templates"),
            algorithm_name=args.algorithm_name,
            num_ranks=args.num_ranks,
            num_steps=args.num_steps
        )
        
        # Generate code
        transpiler = DSLToHcclTranspiler(config)
        generated_files = transpiler.transpile_program(dummy_program)
        
        print(f"Successfully generated HCCL code for {args.algorithm_name}")
        print(f"Collective: {args.collective}, Topology: {args.topology}")
        print(f"Generated files:")
        for file_type, file_path in generated_files.items():
            print(f"  {file_type}: {file_path}")
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())