"""
Utility functions for the RAG Agent system.

This module contains helper functions and utilities used across the RAG agent components.
"""

import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for better compatibility
import matplotlib.pyplot as plt

from agent_visualizer import AgentVisualizer


def create_agent_visualization(save_path: str = "rag_agent_flow.png") -> None:
    """Create a standalone visualization of the RAG agent process flow.
    
    Args:
        save_path: Path to save the generated diagram.
    """
    try:
        print("Creating agent flow diagram...")
        visualizer = AgentVisualizer()
        visualizer.create_agent_flow_diagram(save_path)
        print("Visualization completed successfully!")
    except Exception as e:
        print(f"Error creating visualization: {e}")
        import traceback
        traceback.print_exc()


def test_visualization() -> None:
    """Test function to verify matplotlib is working."""
    try:
        print("Testing matplotlib...")
        plt.figure(figsize=(8, 6))
        plt.plot([1, 2, 3, 4], [1, 4, 2, 3], 'ro-')
        plt.title('Test Plot - Matplotlib is Working!')
        plt.xlabel('X values')
        plt.ylabel('Y values')
        plt.grid(True)
        plt.show()
        print("Matplotlib test successful!")
    except Exception as e:
        print(f"Matplotlib test failed: {e}")
        import traceback
        traceback.print_exc()
