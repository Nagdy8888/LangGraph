"""
Multi-Modal AI Agent Package.

This package provides a comprehensive multi-modal AI agent with various
capabilities including mathematical operations, file management, web search,
text processing, code generation, memory management, and image processing.
"""

from .multi_modal_agent import MultiModalAgent
from .tools import ALL_TOOLS, TOOL_CATEGORIES
from .agent_visualizer import AgentVisualizer, create_agent_visualization
from .utils import (
    ensure_directory_exists, get_file_info, validate_file_type,
    safe_filename, calculate_file_hash, format_file_size,
    create_workspace_structure
)

__version__ = "1.0.0"
__author__ = "Multi-Modal AI Agent Team"
__description__ = "A comprehensive multi-modal AI agent with multiple capabilities"

__all__ = [
    "MultiModalAgent",
    "ALL_TOOLS",
    "TOOL_CATEGORIES", 
    "AgentVisualizer",
    "create_agent_visualization",
    "ensure_directory_exists",
    "get_file_info",
    "validate_file_type",
    "safe_filename",
    "calculate_file_hash",
    "format_file_size",
    "create_workspace_structure"
]
