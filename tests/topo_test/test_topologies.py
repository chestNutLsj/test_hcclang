import os
import sys
import shutil

# Add the parent directory to path to import hcclang
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import networkx as nx
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    HAS_PLOTTING = True
    print("Note: Using non-interactive matplotlib backend (plots will be saved, not displayed)")
except ImportError:
    print("Warning: networkx or matplotlib not installed. Plotting disabled.")
    HAS_PLOTTING = False

from hcclang.topologies import cm384, nvidia, generic

# Create a directory for saving plots
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "topology_plots")
if os.path.exists(PLOTS_DIR):
    shutil.rmtree(PLOTS_DIR)
os.makedirs(PLOTS_DIR)

def draw_topology(topology, title):
    """Draw topology graph if plotting libraries are available."""
    if not HAS_PLOTTING:
        print(f"Plotting disabled for {title}")
        return
        
    G = nx.Graph()
    for i in range(topology.num_nodes()):
        G.add_node(i)
    for i in range(topology.num_nodes()):
        for j in range(i + 1, topology.num_nodes()):
            if topology.links[i][j] > 0:
                G.add_edge(i, j, weight=topology.links[i][j])

    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.title(title)
    # Save plot instead of showing it
    filename = f"topology_{title.replace(' ', '_').lower()}.png"
    filepath = os.path.join(PLOTS_DIR, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"    Plot saved as {filepath}")

def draw_cm384_topology(topology, title, num_npus):
    """
    Custom drawing function for CM384 topology to handle complexity.
    - For > 32 NPUs, it draws an aggregated view of compute nodes.
    - For <= 32 NPUs, it uses a layout better suited for structured graphs.
    """
    if not HAS_PLOTTING:
        print(f"Plotting disabled for {title}")
        return

    G = nx.Graph()
    for i in range(topology.num_nodes()):
        G.add_node(i)
    for i in range(topology.num_nodes()):
        for j in range(i + 1, topology.num_nodes()):
            if topology.links[i][j] > 0:
                G.add_edge(i, j, weight=topology.links[i][j])

    plt.figure(figsize=(12, 12))

    if num_npus > 32:
        # Aggregated view for large topologies (e.g., 128 NPUs)
        # Assuming 8 NPUs per compute node
        num_compute_nodes = num_npus // 8
        aggregated_G = nx.Graph()
        for i in range(num_compute_nodes):
            aggregated_G.add_node(f'Node-{i}')

        for i in range(num_compute_nodes):
            for j in range(i + 1, num_compute_nodes):
                # Calculate aggregated bandwidth between compute nodes
                agg_bw = 0
                for npu1 in range(i * 8, (i + 1) * 8):
                    for npu2 in range(j * 8, (j + 1) * 8):
                        if G.has_edge(npu1, npu2):
                            agg_bw += G[npu1][npu2]['weight']
                if agg_bw > 0:
                    aggregated_G.add_edge(f'Node-{i}', f'Node-{j}', weight=agg_bw)

        pos = nx.spring_layout(aggregated_G, seed=42)
        nx.draw(aggregated_G, pos, with_labels=True, node_color='skyblue', node_size=2000, font_size=10)
        labels = nx.get_edge_attributes(aggregated_G, 'weight')
        nx.draw_networkx_edge_labels(aggregated_G, pos, edge_labels=labels, font_color='red')
        plt.title(f"{title} (Aggregated View)")
    else:
        # Detailed view for smaller topologies with a better layout
        try:
            pos = nx.kamada_kawai_layout(G)
        except ImportError:
            # Fallback to spring layout if scipy is not available
            print("    Note: Using spring layout (install scipy for better structured layout)")
            pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, node_color='lightgreen', node_size=500, font_size=10)
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        plt.title(title)

    filename = f"topology_{title.replace(' ', '_').lower()}.png"
    filepath = os.path.join(PLOTS_DIR, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"    Custom plot for CM384 saved as {filepath}")

def print_topology_info(topology, name):
    """Print basic topology information."""
    print(f"\n{name} Topology Info:")
    print(f"  - Name: {topology.name}")
    print(f"  - Nodes: {topology.num_nodes()}")
    print(f"  - Links matrix size: {len(topology.links)}x{len(topology.links[0]) if topology.links else 0}")
    
    # Print some sample connections
    if topology.num_nodes() > 0:
        print(f"  - Sample connections:")
        connections_shown = 0
        for i in range(min(4, topology.num_nodes())):
            for j in range(i + 1, min(4, topology.num_nodes())):
                if topology.links[i][j] > 0:
                    print(f"    Node {i} -> Node {j}: {topology.links[i][j]} GB/s")
                    connections_shown += 1
                    if connections_shown >= 3:
                        break
            if connections_shown >= 3:
                break

def test_cm384_topology():
    """Test CM384 topology."""
    print("="*50)
    print("Testing CM384 Topology")
    print("="*50)
    
    try:
        # Test different sizes
        for num_npus in [8, 16, 32, 128]:
            print(f"\nTesting CM384 with {num_npus} NPUs...")
            cm384_topo = cm384.cm384(num_npus)
            print_topology_info(cm384_topo, f"CM384-{num_npus}")
            # Use the custom drawing function for CM384
            draw_cm384_topology(cm384_topo, f"CM384 {num_npus} NPUs", num_npus)
            
    except Exception as e:
        print(f"❌ CM384 topology test failed: {e}")
        import traceback
        traceback.print_exc()

def test_nvidia_topologies():
    """Test NVIDIA topologies."""
    print("="*50)
    print("Testing NVIDIA Topologies")
    print("="*50)
    
    try:
        # Test available NVIDIA topologies
        available_funcs = ['dgx_a100', 'dgx_superpod']
        
        print(f"Available NVIDIA functions: {available_funcs}")
        
        for func_name in available_funcs:
            try:
                print(f"\nTesting NVIDIA {func_name}...")
                topo_func = getattr(nvidia, func_name)
                if func_name == 'dgx_superpod':
                    # Test dgx_superpod with default parameters
                    topo = topo_func()
                else:
                    topo = topo_func()
                print_topology_info(topo, f"NVIDIA-{func_name}")
                draw_topology(topo, f"NVIDIA {func_name}")
            except Exception as e:
                print(f"⚠️  Failed to test {func_name}: {e}")
                
    except Exception as e:
        print(f"❌ NVIDIA topology test failed: {e}")
        import traceback
        traceback.print_exc()

def test_generic_topologies():
    """Test generic topologies."""
    print("="*50)
    print("Testing Generic Topologies")
    print("="*50)
    
    try:
        # Test basic topologies with fixed parameters
        test_cases = [
            ('ring', lambda: generic.ring(8), "8-node ring"),
            ('line', lambda: generic.line(6), "6-node line"),
            ('star', lambda: generic.star(5), "5-node star"),
            ('fully_connected', lambda: generic.fully_connected(4), "4-node fully connected"),
        ]
        
        for name, func, description in test_cases:
            try:
                print(f"\nTesting {description}...")
                topo = func()
                print_topology_info(topo, name)
                draw_topology(topo, f"Generic {description}")
            except Exception as e:
                print(f"⚠️  Failed to create {name}: {e}")
                
    except Exception as e:
        print(f"❌ Generic topology test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("HCCLang Topology Testing Suite")
    print("=" * 60)
    
    # Run all tests
    test_cm384_topology()
    test_nvidia_topologies()
    test_generic_topologies()
    
    print("\n" + "="*60)
    print("Topology testing completed!")
