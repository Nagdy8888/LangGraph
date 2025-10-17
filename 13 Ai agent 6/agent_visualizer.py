"""
Agent Visualizer Module.

This module provides visualization capabilities for the multi-modal agent,
including flow diagrams and process visualization.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np
from typing import Dict, List, Tuple, Optional

from config import (
    GRAPH_WIDTH, GRAPH_HEIGHT, NODE_COLORS, EDGE_COLORS,
    WORKSPACE_DIR
)


class AgentVisualizer:
    """Visualizer for the multi-modal agent flow and processes."""
    
    def __init__(self):
        """Initialize the agent visualizer."""
        self.fig = None
        self.ax = None
        self.node_positions = {}
        self.node_connections = []
    
    def create_agent_flow_diagram(self, save_path: str = "multi_modal_agent_flow.png") -> None:
        """Create a visual diagram of the agent's process flow.
        
        Args:
            save_path: Path to save the generated diagram.
        """
        try:
            self._setup_figure()
            self._create_flow_diagram()
            self._save_diagram(save_path)
            print(f"Agent flow diagram saved to: {save_path}")
        except Exception as e:
            print(f"Error creating flow diagram: {e}")
    
    def _setup_figure(self) -> None:
        """Setup the matplotlib figure and axes."""
        plt.style.use('default')
        self.fig, self.ax = plt.subplots(figsize=(GRAPH_WIDTH, GRAPH_HEIGHT))
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 8)
        self.ax.axis('off')
        self.fig.patch.set_facecolor('#f0f0f0')
        self.ax.set_facecolor('#f8f8f8')
    
    def _create_flow_diagram(self) -> None:
        """Create the main flow diagram."""
        # Define node positions
        self.node_positions = {
            'start': (1, 6.5),
            'agent': (3, 6.5),
            'tools': (5, 6.5),
            'calculator': (2, 4.5),
            'file_manager': (4, 4.5),
            'web_search': (6, 4.5),
            'text_processor': (2, 2.5),
            'code_processor': (4, 2.5),
            'memory_manager': (6, 2.5),
            'image_processor': (4, 0.5),
            'end': (8, 6.5)
        }
        
        # Create nodes
        self._create_nodes()
        
        # Create connections
        self._create_connections()
        
        # Add title and legend
        self._add_title_and_legend()
    
    def _create_nodes(self) -> None:
        """Create all nodes in the diagram."""
        # Start node
        self._create_node('start', 'START', NODE_COLORS['start'], 'oval')
        
        # Main agent node
        self._create_node('agent', 'Multi-Modal\nAgent', NODE_COLORS['agent'], 'rect')
        
        # Tools node
        self._create_node('tools', 'Tool\nExecution', NODE_COLORS['agent'], 'rect')
        
        # Tool category nodes
        self._create_node('calculator', 'Calculator\nTools', NODE_COLORS['calculator'], 'rect')
        self._create_node('file_manager', 'File\nOperations', NODE_COLORS['file_manager'], 'rect')
        self._create_node('web_search', 'Web\nSearch', NODE_COLORS['web_search'], 'rect')
        self._create_node('text_processor', 'Text\nProcessing', NODE_COLORS['text_processor'], 'rect')
        self._create_node('code_processor', 'Code\nProcessing', NODE_COLORS['code_processor'], 'rect')
        self._create_node('memory_manager', 'Memory\nManagement', NODE_COLORS['memory_manager'], 'rect')
        self._create_node('image_processor', 'Image\nProcessing', NODE_COLORS['image_processor'], 'rect')
        
        # End node
        self._create_node('end', 'END', NODE_COLORS['end'], 'oval')
    
    def _create_node(self, node_id: str, label: str, color: str, shape: str) -> None:
        """Create a single node in the diagram.
        
        Args:
            node_id: Unique identifier for the node
            label: Text label for the node
            color: Color of the node
            shape: Shape type ('rect', 'oval', 'diamond')
        """
        x, y = self.node_positions[node_id]
        
        if shape == 'oval':
            width, height = 0.8, 0.6
            bbox = FancyBboxPatch(
                (x - width/2, y - height/2), width, height,
                boxstyle="round,pad=0.1",
                facecolor=color,
                edgecolor='black',
                linewidth=2,
                alpha=0.8
            )
        elif shape == 'diamond':
            width, height = 0.8, 0.6
            bbox = mpatches.Polygon([
                (x, y + height/2), (x + width/2, y),
                (x, y - height/2), (x - width/2, y)
            ], facecolor=color, edgecolor='black', linewidth=2, alpha=0.8)
        else:  # rect
            width, height = 1.2, 0.8
            bbox = FancyBboxPatch(
                (x - width/2, y - height/2), width, height,
                boxstyle="round,pad=0.1",
                facecolor=color,
                edgecolor='black',
                linewidth=2,
                alpha=0.8
            )
        
        self.ax.add_patch(bbox)
        self.ax.text(x, y, label, ha='center', va='center', 
                    fontsize=10, fontweight='bold', color='white')
    
    def _create_connections(self) -> None:
        """Create connections between nodes."""
        # Main flow
        self._create_arrow('start', 'agent', EDGE_COLORS['normal'])
        self._create_arrow('agent', 'tools', EDGE_COLORS['tool_call'])
        self._create_arrow('tools', 'agent', EDGE_COLORS['normal'])
        self._create_arrow('agent', 'end', EDGE_COLORS['normal'])
        
        # Tool category connections
        self._create_arrow('tools', 'calculator', EDGE_COLORS['conditional'])
        self._create_arrow('tools', 'file_manager', EDGE_COLORS['conditional'])
        self._create_arrow('tools', 'web_search', EDGE_COLORS['conditional'])
        self._create_arrow('tools', 'text_processor', EDGE_COLORS['conditional'])
        self._create_arrow('tools', 'code_processor', EDGE_COLORS['conditional'])
        self._create_arrow('tools', 'memory_manager', EDGE_COLORS['conditional'])
        self._create_arrow('tools', 'image_processor', EDGE_COLORS['conditional'])
        
        # Return arrows from tool categories
        self._create_arrow('calculator', 'tools', EDGE_COLORS['normal'], style='dashed')
        self._create_arrow('file_manager', 'tools', EDGE_COLORS['normal'], style='dashed')
        self._create_arrow('web_search', 'tools', EDGE_COLORS['normal'], style='dashed')
        self._create_arrow('text_processor', 'tools', EDGE_COLORS['normal'], style='dashed')
        self._create_arrow('code_processor', 'tools', EDGE_COLORS['normal'], style='dashed')
        self._create_arrow('memory_manager', 'tools', EDGE_COLORS['normal'], style='dashed')
        self._create_arrow('image_processor', 'tools', EDGE_COLORS['normal'], style='dashed')
    
    def _create_arrow(self, from_node: str, to_node: str, color: str, style: str = 'solid') -> None:
        """Create an arrow connection between two nodes.
        
        Args:
            from_node: Source node ID
            to_node: Target node ID
            color: Arrow color
            style: Arrow style ('solid', 'dashed', 'dotted')
        """
        if from_node not in self.node_positions or to_node not in self.node_positions:
            return
        
        x1, y1 = self.node_positions[from_node]
        x2, y2 = self.node_positions[to_node]
        
        # Calculate arrow properties
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        
        if length == 0:
            return
        
        # Normalize direction
        dx_norm = dx / length
        dy_norm = dy / length
        
        # Adjust start and end points to node edges
        offset = 0.4
        x1 += dx_norm * offset
        y1 += dy_norm * offset
        x2 -= dx_norm * offset
        y2 -= dy_norm * offset
        
        # Create arrow
        arrow = mpatches.FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle='->',
            mutation_scale=20,
            color=color,
            linewidth=2,
            linestyle=style,
            alpha=0.8
        )
        
        self.ax.add_patch(arrow)
    
    def _add_title_and_legend(self) -> None:
        """Add title and legend to the diagram."""
        # Title
        self.ax.text(5, 7.5, 'Multi-Modal AI Agent Flow Diagram', 
                    ha='center', va='center', fontsize=16, fontweight='bold')
        
        # Legend
        legend_elements = [
            mpatches.Patch(color=NODE_COLORS['agent'], label='Main Agent'),
            mpatches.Patch(color=NODE_COLORS['calculator'], label='Calculator Tools'),
            mpatches.Patch(color=NODE_COLORS['file_manager'], label='File Operations'),
            mpatches.Patch(color=NODE_COLORS['web_search'], label='Web Search'),
            mpatches.Patch(color=NODE_COLORS['text_processor'], label='Text Processing'),
            mpatches.Patch(color=NODE_COLORS['code_processor'], label='Code Processing'),
            mpatches.Patch(color=NODE_COLORS['memory_manager'], label='Memory Management'),
            mpatches.Patch(color=NODE_COLORS['image_processor'], label='Image Processing'),
        ]
        
        self.ax.legend(handles=legend_elements, loc='upper right', 
                      bbox_to_anchor=(0.98, 0.98), fontsize=8)
        
        # Add description
        description = (
            "This diagram shows the flow of the Multi-Modal AI Agent.\n"
            "The agent processes user input and can execute various tools\n"
            "based on the request type and requirements."
        )
        self.ax.text(5, -0.5, description, ha='center', va='center', 
                    fontsize=10, style='italic', color='gray')
    
    def _save_diagram(self, save_path: str) -> None:
        """Save the diagram to file.
        
        Args:
            save_path: Path to save the diagram.
        """
        full_path = f"{WORKSPACE_DIR}/{save_path}"
        plt.tight_layout()
        plt.savefig(full_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
    
    def create_tool_usage_chart(self, tool_usage_data: Dict[str, int], 
                               save_path: str = "tool_usage_chart.png") -> None:
        """Create a chart showing tool usage statistics.
        
        Args:
            tool_usage_data: Dictionary with tool names and usage counts
            save_path: Path to save the chart
        """
        try:
            if not tool_usage_data:
                print("No tool usage data provided")
                return
            
            self.fig, self.ax = plt.subplots(figsize=(12, 8))
            
            tools = list(tool_usage_data.keys())
            counts = list(tool_usage_data.values())
            
            # Create bar chart
            bars = self.ax.bar(tools, counts, color=[NODE_COLORS.get(tool, '#666666') for tool in tools])
            
            # Customize chart
            self.ax.set_title('Tool Usage Statistics', fontsize=16, fontweight='bold')
            self.ax.set_xlabel('Tools', fontsize=12)
            self.ax.set_ylabel('Usage Count', fontsize=12)
            self.ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                self.ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                           f'{count}', ha='center', va='bottom')
            
            plt.tight_layout()
            full_path = f"{WORKSPACE_DIR}/{save_path}"
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Tool usage chart saved to: {save_path}")
            
        except Exception as e:
            print(f"Error creating tool usage chart: {e}")
    
    def create_conversation_timeline(self, conversation_data: List[Dict], 
                                   save_path: str = "conversation_timeline.png") -> None:
        """Create a timeline visualization of conversation.
        
        Args:
            conversation_data: List of conversation entries
            save_path: Path to save the timeline
        """
        try:
            if not conversation_data:
                print("No conversation data provided")
                return
            
            self.fig, self.ax = plt.subplots(figsize=(14, 8))
            
            # Process conversation data
            timestamps = []
            roles = []
            lengths = []
            
            for entry in conversation_data:
                timestamps.append(entry.get('timestamp', ''))
                roles.append(entry.get('role', 'unknown'))
                lengths.append(len(entry.get('content', '')))
            
            # Create timeline
            y_pos = np.arange(len(conversation_data))
            colors = ['#4CAF50' if role == 'user' else '#2196F3' for role in roles]
            
            bars = self.ax.barh(y_pos, lengths, color=colors, alpha=0.7)
            
            # Customize chart
            self.ax.set_title('Conversation Timeline', fontsize=16, fontweight='bold')
            self.ax.set_xlabel('Message Length (characters)', fontsize=12)
            self.ax.set_ylabel('Message Index', fontsize=12)
            
            # Add legend
            user_patch = mpatches.Patch(color='#4CAF50', label='User Messages')
            assistant_patch = mpatches.Patch(color='#2196F3', label='Assistant Messages')
            self.ax.legend(handles=[user_patch, assistant_patch])
            
            plt.tight_layout()
            full_path = f"{WORKSPACE_DIR}/{save_path}"
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Conversation timeline saved to: {save_path}")
            
        except Exception as e:
            print(f"Error creating conversation timeline: {e}")


def create_agent_visualization() -> None:
    """Create a complete agent visualization."""
    visualizer = AgentVisualizer()
    visualizer.create_agent_flow_diagram()


def create_tool_usage_visualization(tool_usage_data: Dict[str, int]) -> None:
    """Create tool usage visualization.
    
    Args:
        tool_usage_data: Dictionary with tool names and usage counts
    """
    visualizer = AgentVisualizer()
    visualizer.create_tool_usage_chart(tool_usage_data)


def create_conversation_visualization(conversation_data: List[Dict]) -> None:
    """Create conversation visualization.
    
    Args:
        conversation_data: List of conversation entries
    """
    visualizer = AgentVisualizer()
    visualizer.create_conversation_timeline(conversation_data)
