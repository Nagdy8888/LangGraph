"""
Agent visualization module for the RAG Agent.

This module handles visualization of the RAG agent's process flow.
"""

import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for better compatibility
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

from config import (
    GRAPH_WIDTH, GRAPH_HEIGHT, NODE_COLORS, EDGE_COLORS
)


class AgentVisualizer:
    """Handles visualization of the RAG agent's process flow."""
    
    def __init__(self):
        """Initialize the visualizer with matplotlib settings."""
        plt.style.use('default')
        self.fig = None
        self.ax = None
    
    def create_agent_flow_diagram(self, save_path: str = "rag_agent_flow.png") -> None:
        """Create a visual diagram of the RAG agent's process flow.
        
        Args:
            save_path: Path to save the generated diagram.
        """
        self.fig, self.ax = plt.subplots(figsize=(GRAPH_WIDTH, GRAPH_HEIGHT))
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 8)
        self.ax.axis('off')
        
        # Define node positions
        nodes = {
            'start': (1, 6),
            'llm': (3, 6),
            'retriever_agent': (7, 6),
            'end': (9, 6),
            'decision': (5, 4)
        }
        
        # Draw nodes
        self._draw_node('start', nodes['start'], 'START', NODE_COLORS['start'])
        self._draw_node('llm', nodes['llm'], 'LLM\nAgent', NODE_COLORS['llm'])
        self._draw_node('retriever_agent', nodes['retriever_agent'], 'Retriever\nAgent', NODE_COLORS['retriever_agent'])
        self._draw_node('end', nodes['end'], 'END', NODE_COLORS['end'])
        self._draw_decision_diamond(nodes['decision'])
        
        # Draw edges
        self._draw_arrow(nodes['start'], nodes['llm'], 'User Question')
        self._draw_conditional_arrow(nodes['llm'], nodes['decision'], 'Has Tool Calls?', EDGE_COLORS['conditional'])
        self._draw_arrow(nodes['decision'], nodes['retriever_agent'], 'Yes', EDGE_COLORS['conditional'])
        self._draw_arrow(nodes['decision'], nodes['end'], 'No', EDGE_COLORS['conditional'])
        self._draw_arrow(nodes['retriever_agent'], nodes['llm'], 'Tool Results')
        
        # Add title and description
        self.ax.text(5, 7.5, 'RAG Agent Process Flow', 
                    fontsize=16, fontweight='bold', ha='center')
        
        # Add legend
        self._add_legend()
        
        # Add process description
        self._add_process_description()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Agent flow diagram saved to: {save_path}")
        plt.show()
    
    def _draw_node(self, node_id: str, position: tuple, label: str, color: str) -> None:
        """Draw a rectangular node.
        
        Args:
            node_id: Unique identifier for the node.
            position: (x, y) coordinates for the node center.
            label: Text label for the node.
            color: Color of the node.
        """
        x, y = position
        width, height = 1.2, 0.8
        
        # Create rounded rectangle
        rect = FancyBboxPatch(
            (x - width/2, y - height/2), width, height,
            boxstyle="round,pad=0.1",
            facecolor=color,
            edgecolor='black',
            linewidth=2
        )
        self.ax.add_patch(rect)
        
        # Add label
        self.ax.text(x, y, label, ha='center', va='center', 
                    fontsize=10, fontweight='bold', color='white')
    
    def _draw_decision_diamond(self, position: tuple) -> None:
        """Draw a decision diamond.
        
        Args:
            position: (x, y) coordinates for the diamond center.
        """
        x, y = position
        size = 0.6
        
        # Create diamond shape
        diamond = patches.RegularPolygon(
            (x, y), 4, radius=size,
            orientation=np.pi/4,
            facecolor='#FFC107',
            edgecolor='black',
            linewidth=2
        )
        self.ax.add_patch(diamond)
        
        # Add label
        self.ax.text(x, y, 'Decision', ha='center', va='center', 
                    fontsize=9, fontweight='bold')
    
    def _draw_arrow(self, start: tuple, end: tuple, label: str = "", color: str = EDGE_COLORS['normal']) -> None:
        """Draw an arrow between two points.
        
        Args:
            start: Starting (x, y) coordinates.
            end: Ending (x, y) coordinates.
            label: Label for the arrow.
            color: Color of the arrow.
        """
        # Calculate arrow properties
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = np.sqrt(dx**2 + dy**2)
        
        # Normalize direction
        dx_norm = dx / length
        dy_norm = dy / length
        
        # Adjust start and end points to avoid overlapping with nodes
        start_adj = (start[0] + dx_norm * 0.6, start[1] + dy_norm * 0.4)
        end_adj = (end[0] - dx_norm * 0.6, end[1] - dy_norm * 0.4)
        
        # Draw arrow
        arrow = patches.FancyArrowPatch(
            start_adj, end_adj,
            arrowstyle='->',
            mutation_scale=20,
            color=color,
            linewidth=2
        )
        self.ax.add_patch(arrow)
        
        # Add label
        if label:
            mid_x = (start_adj[0] + end_adj[0]) / 2
            mid_y = (start_adj[1] + end_adj[1]) / 2
            self.ax.text(mid_x, mid_y + 0.2, label, ha='center', va='bottom', 
                        fontsize=8, color=color, fontweight='bold')
    
    def _draw_conditional_arrow(self, start: tuple, end: tuple, label: str, color: str) -> None:
        """Draw a conditional arrow with special styling.
        
        Args:
            start: Starting (x, y) coordinates.
            end: Ending (x, y) coordinates.
            label: Label for the arrow.
            color: Color of the arrow.
        """
        self._draw_arrow(start, end, label, color)
    
    def _add_legend(self) -> None:
        """Add a legend explaining the diagram elements."""
        legend_elements = [
            patches.Patch(color=NODE_COLORS['start'], label='Start Node'),
            patches.Patch(color=NODE_COLORS['llm'], label='LLM Processing'),
            patches.Patch(color=NODE_COLORS['retriever_agent'], label='Tool Execution'),
            patches.Patch(color=NODE_COLORS['end'], label='End Node'),
            patches.Patch(color='#FFC107', label='Decision Point'),
            patches.Patch(color=EDGE_COLORS['normal'], label='Normal Flow'),
            patches.Patch(color=EDGE_COLORS['conditional'], label='Conditional Flow')
        ]
        
        self.ax.legend(handles=legend_elements, loc='upper right', 
                      bbox_to_anchor=(0.98, 0.98), fontsize=9)
    
    def _add_process_description(self) -> None:
        """Add a description of the agent process."""
        description = """
Process Description:
1. User asks a question
2. LLM Agent processes the question
3. If tool calls are needed, go to Retriever Agent
4. Retriever Agent searches documents and returns results
5. LLM Agent processes results and provides final answer
6. Process ends when no more tool calls are needed
        """
        
        self.ax.text(0.5, 1.5, description, fontsize=9, ha='left', va='top',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
