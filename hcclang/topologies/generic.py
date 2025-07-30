# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .topo_tools import Topology, compose_rings, compose_hierarchical_rings

def hub_and_spoke(num_nodes):
    links = [[0 if x==y else 1 for y in range(num_nodes)] for x in range(num_nodes)]
    switches = []
    for node in range(num_nodes):
        others = [other for other in range(num_nodes) if other != node]
        switches.append(([node],others,1,f'node_{node}_out'))
        switches.append((others,[node],1,f'node_{node}_in'))
    return Topology(f'HubAndSpoke(n={num_nodes})', links, switches)

def fully_connected(num_nodes):
    links = []
    for i in range(num_nodes):
        row = [1] * num_nodes
        row[i] = 0
        links.append(row)
    return Topology(f'FullyConnected(n={num_nodes})', links)

def ring(num_nodes):
    links = []
    for i in range(num_nodes):
        row = [0] * num_nodes
        row[(i+1) % num_nodes] = 1
        row[(i-1) % num_nodes] = 1
        links.append(row)
    return Topology(f'Ring(n={num_nodes})', links)

def line(num_nodes):
    links = []
    for i in range(num_nodes):
        row = [0] * num_nodes
        if i - 1 >= 0:
            row[i-1] = 1
        if i + 1 < num_nodes:
            row[i+1] = 1
        links.append(row)
    return Topology(f'Line(n={num_nodes})', links)

def star(num_nodes, non_blocking=True):
    links = [[0 if i == 0 else 1 for i in range(num_nodes)]]
    for i in range(1, num_nodes):
        links.append([1 if j == 0 else 0 for j in range(num_nodes)])
    switches = []
    if not non_blocking:
        points = [i for i in range(num_nodes) if i != 0]
        switches.append(([0],points,1,f'to_points'))
        switches.append((points,[0],1,f'from_points'))
    return Topology(f'Star(n={num_nodes})', links, switches)

def multi_ring(num_nodes, num_rings):
    """
    Create a multi-ring topology by composing multiple single rings.
    
    Args:
        num_nodes: Total number of nodes
        num_rings: Number of parallel rings
        
    Returns:
        Topology object representing multi-ring topology
        
    This function reuses the existing ring() function to create multiple
    parallel rings, demonstrating composition-based topology design.
    """
    if num_nodes % num_rings != 0:
        raise ValueError(f"num_nodes ({num_nodes}) must be divisible by num_rings ({num_rings})")
    
    nodes_per_ring = num_nodes // num_rings
    
    # Create individual ring topologies and compose them
    return compose_rings(num_rings, nodes_per_ring)

def hierarchical_ring(num_nodes, nodes_per_level):
    """
    Create a hierarchical ring topology by composing rings with inter-group connections.
    
    Args:
        num_nodes: Total number of nodes
        nodes_per_level: Number of nodes at each level
        
    Returns:
        Topology object representing hierarchical ring topology
        
    This function reuses the existing ring() function and composition utilities
    to create hierarchical ring structures.
    """
    if num_nodes % nodes_per_level != 0:
        raise ValueError(f"num_nodes ({num_nodes}) must be divisible by nodes_per_level ({nodes_per_level})")
    
    num_groups = num_nodes // nodes_per_level
    
    # Use composition-based approach
    return compose_hierarchical_rings(num_groups, nodes_per_level)
