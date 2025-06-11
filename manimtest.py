from manim import *
import networkx as nx
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors

class NetworkXGraphScene(Scene):
    def construct(self):
        # Create a random graph
        G = nx.erdos_renyi_graph(n=10, p=0.3, seed=42)
        
        # Generate initial random phases for nodes
        phases = {node: np.random.uniform(0, 2 * np.pi) for node in G.nodes}
        
        # Normalize phases to [0,1] for color mapping
        norm = mcolors.Normalize(vmin=0, vmax=2 * np.pi)
        colormap = cm.hsv  # Use HSV colormap to reflect phase as hue
        
        # Assign colors based on phases
        def get_colors():
            return {
                node: rgb_to_color(colormap(norm(phases[node]))[:3]) for node in G.nodes
            }
        
        # Create Manim graph
        graph = Graph(
            list(G.nodes),
            list(G.edges),
            layout="spring",
            vertex_config={node: {"fill_color": get_colors()[node]} for node in G.nodes},
            edge_config={"stroke_color": WHITE}
        )
        
        self.play(Create(graph))
        self.wait(1)
        
        # Animate phase changes and node movement
        for _ in range(5):
            for node in G.nodes:
                phases[node] += np.pi / 4  # Increment phase
                phases[node] %= 2 * np.pi  # Keep in range [0, 2pi]
            
            new_colors = {node: {"fill_color": get_colors()[node]} for node in G.nodes}
            new_layout = {node: np.append(pos, [0]) for node, pos in nx.spring_layout(G, seed=42).items()}

            
            self.play(
                *[graph.animate.change_layout(new_layout)],
                *[graph.animate.set_vertex_attributes({node: new_colors[node]}) for node in G.nodes],
                run_time=1, rate_func=smooth
            )
        
        self.wait(2)
