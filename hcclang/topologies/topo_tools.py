# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

class Topology(object):
    def __init__(self, name, links, switches=[]):
        self.name = name
        self.links = links
        self.switches = switches
        for srcs, dsts, bw, switch_name in switches:
            if bw == 0:
                raise ValueError(f'Switch {switch_name} has zero bandwidth, but switch bandwidths must be strictly positive. Please encode connectedness in links.')
            if bw < 0:
                raise ValueError(f'Switch {switch_name} has a negative bandwidth of {bw}. Bandwidth must be strictly positive.')

    def sources(self, dst):
        for src, bw in enumerate(self.links[dst]):
            if bw > 0:
                yield src

    def destinations(self, src):
        for dst, links in enumerate(self.links):
            bw = links[src]
            if bw > 0:
                yield dst

    def link(self, src, dst):
        return self.links[dst][src]

    def num_nodes(self):
        return len(self.links)

    def nodes(self):
        return range(self.num_nodes())
    
    def bandwidth_constraints(self):
        for dst, dst_links in enumerate(self.links):
            for src, bw in enumerate(dst_links):
                if bw > 0:
                    yield ([src], [dst], bw, f'{src}→{dst}')
        for srcs, dsts, bw, switch_name in self.switches:
            yield (srcs, dsts, bw, switch_name)


# Topology transformation utilities
def reverse_topology(topology):
    '''
    Reverses the direction of all links and switches in the topology.
    '''
    num_nodes = topology.num_nodes()
    # Transpose the links
    links = [[topology.links[src][dst] for src in range(num_nodes)] for dst in range(num_nodes)]
    # Reverse the switches
    switches = [(dsts, srcs, bw, f'{name}_reversed') for srcs, dsts, bw, name in topology.switches]
    return Topology(f'Reverse{topology.name}', links, switches)

def binarize_topology(topology):
    '''
    Makes all link bandwidths 1 and removes all switches. Essentially, the bandwidth modeling part of the topology
    is stripped out and only connectivity information is kept.
    '''
    num_nodes = topology.num_nodes()
    links = [[1 if topology.links[src][dst] > 0 else 0 for src in range(num_nodes)] for dst in range(num_nodes)]
    return Topology(f'Binarized{topology.name}', links, [])

# Topology composition utilities
def map_node_ids(topology, node_mapping):
    """
    Remap node IDs in a topology according to the given mapping.
    
    Args:
        topology: Source topology
        node_mapping: Dict mapping old node IDs to new node IDs
        
    Returns:
        New topology with remapped node IDs
    """
    num_nodes = max(node_mapping.values()) + 1
    links = [[0 for _ in range(num_nodes)] for _ in range(num_nodes)]
    
    # Remap links
    for old_dst in range(topology.num_nodes()):
        for old_src in range(topology.num_nodes()):
            if topology.links[old_dst][old_src] > 0:
                new_dst = node_mapping[old_dst]
                new_src = node_mapping[old_src]
                links[new_dst][new_src] = topology.links[old_dst][old_src]
    
    # Remap switches
    switches = []
    for srcs, dsts, bw, name in topology.switches:
        new_srcs = [node_mapping[src] for src in srcs]
        new_dsts = [node_mapping[dst] for dst in dsts]
        switches.append((new_srcs, new_dsts, bw, name))
    
    return Topology(f'Mapped{topology.name}', links, switches)

def compose_topologies(*topologies):
    """
    Compose multiple topologies into a single topology by combining their node spaces.
    
    Args:
        *topologies: Variable number of topology objects to compose
        
    Returns:
        Composed topology with all input topologies as disjoint components
    """
    if not topologies:
        return Topology('Empty', [], [])
    
    total_nodes = sum(topo.num_nodes() for topo in topologies)
    links = [[0 for _ in range(total_nodes)] for _ in range(total_nodes)]
    switches = []
    
    node_offset = 0
    names = []
    
    for i, topo in enumerate(topologies):
        names.append(topo.name)
        num_nodes = topo.num_nodes()
        
        # Copy links with offset
        for dst in range(num_nodes):
            for src in range(num_nodes):
                if topo.links[dst][src] > 0:
                    new_dst = dst + node_offset
                    new_src = src + node_offset
                    links[new_dst][new_src] = topo.links[dst][src]
        
        # Copy switches with offset
        for srcs, dsts, bw, name in topo.switches:
            new_srcs = [src + node_offset for src in srcs]
            new_dsts = [dst + node_offset for dst in dsts]
            switches.append((new_srcs, new_dsts, bw, f'{name}_topo{i}'))
        
        node_offset += num_nodes
    
    composed_name = f'Composed({"+".join(names)})'
    return Topology(composed_name, links, switches)

def compose_rings(num_rings, nodes_per_ring):
    """
    Compose multiple ring topologies into a multi-ring topology.
    
    Args:
        num_rings: Number of rings to create
        nodes_per_ring: Number of nodes in each ring
        
    Returns:
        Multi-ring topology composed of individual rings
    """
    # Import here to avoid circular dependency
    from .generic import ring
    
    # Create individual ring topologies
    individual_rings = []
    for ring_id in range(num_rings):
        single_ring = ring(nodes_per_ring)
        # Rename to include ring ID
        single_ring.name = f'Ring{ring_id}(n={nodes_per_ring})'
        individual_rings.append(single_ring)
    
    # Compose all rings into a single topology
    multi_ring_topo = compose_topologies(*individual_rings)
    multi_ring_topo.name = f'MultiRing(n={num_rings * nodes_per_ring},rings={num_rings})'
    
    return multi_ring_topo

def compose_hierarchical_rings(num_groups, nodes_per_group):
    """
    Compose hierarchical ring topology by creating local rings and inter-group connections.
    
    Args:
        num_groups: Number of groups (levels)
        nodes_per_group: Number of nodes in each group
        
    Returns:
        Hierarchical ring topology with intra-group and inter-group connections
    """
    # Import here to avoid circular dependency
    from .generic import ring
    
    total_nodes = num_groups * nodes_per_group
    
    # Start with individual rings for each group
    group_rings = []
    for group_id in range(num_groups):
        group_ring = ring(nodes_per_group)
        group_ring.name = f'Group{group_id}Ring(n={nodes_per_group})'
        group_rings.append(group_ring)
    
    # Compose groups into base topology
    base_topo = compose_topologies(*group_rings)
    
    # Add inter-group connections
    links = [row[:] for row in base_topo.links]  # Deep copy
    switches = list(base_topo.switches)  # Copy switches
    
    # Add inter-group connections between corresponding positions
    for pos in range(nodes_per_group):
        for group_id in range(num_groups):
            curr_node = group_id * nodes_per_group + pos
            next_group = (group_id + 1) % num_groups
            prev_group = (group_id - 1 + num_groups) % num_groups
            
            next_node = next_group * nodes_per_group + pos
            prev_node = prev_group * nodes_per_group + pos
            
            # Add bidirectional inter-group connections
            links[curr_node][next_node] = 1
            links[curr_node][prev_node] = 1
            
            # Add inter-group switches
            switches.append(([curr_node], [next_node], 1, f'intergroup_{group_id}_to_{next_group}_pos_{pos}'))
    
    return Topology(f'HierarchicalRing(n={total_nodes},groups={num_groups})', links, switches)
