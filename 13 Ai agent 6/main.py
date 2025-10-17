"""
Main entry point for the Multi-Modal AI Agent.

This module provides the main entry point and interactive session management
for the multi-modal agent system.
"""

import sys
import os
import argparse
from pathlib import Path
from dotenv import load_dotenv

from multi_modal_agent import MultiModalAgent
from agent_visualizer import create_agent_visualization, create_tool_usage_visualization
from utils import create_workspace_structure, load_config_file, save_config_file


def setup_environment() -> bool:
    """Setup the environment and workspace.
    
    Returns:
        True if setup successful
    """
    try:
        # Load environment variables
        load_dotenv()
        
        # Create workspace structure
        if not create_workspace_structure():
            print("Warning: Could not create workspace structure")
        
        # Load configuration
        config = load_config_file()
        
        # Check for required environment variables
        required_vars = ['OPENAI_API_KEY']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
            print("Please set these variables in your .env file or environment")
            return False
        
        return True
    
    except Exception as e:
        print(f"Error setting up environment: {e}")
        return False


def run_interactive_mode() -> None:
    """Run the agent in interactive mode."""
    try:
        print("Initializing Multi-Modal AI Agent...")
        agent = MultiModalAgent()
        print("Agent initialized successfully!")
        
        agent.run_interactive_session()
        
    except KeyboardInterrupt:
        print("\n\nSession interrupted by user.")
    except Exception as e:
        print(f"Error running interactive mode: {e}")
        import traceback
        traceback.print_exc()


def run_single_query(query: str) -> None:
    """Run a single query and display the result.
    
    Args:
        query: Query to process
    """
    try:
        print("Initializing Multi-Modal AI Agent...")
        agent = MultiModalAgent()
        print("Agent initialized successfully!")
        
        print(f"\nProcessing query: {query}")
        response = agent.process_message(query)
        
        print("\n=== RESPONSE ===")
        print(response)
        
    except Exception as e:
        print(f"Error processing query: {e}")
        import traceback
        traceback.print_exc()


def create_visualization() -> None:
    """Create agent flow visualization."""
    try:
        print("Creating agent flow diagram...")
        create_agent_visualization()
        print("Visualization created successfully!")
        
    except Exception as e:
        print(f"Error creating visualization: {e}")


def show_help() -> None:
    """Show help information."""
    help_text = """
Multi-Modal AI Agent - Help

Usage:
    python main.py [options]

Options:
    -h, --help              Show this help message
    -q, --query QUERY       Process a single query and exit
    -v, --visualize         Create agent flow diagram
    -c, --config            Show current configuration
    -i, --interactive       Run in interactive mode (default)

Examples:
    python main.py                           # Run interactive mode
    python main.py -q "What is 2+2?"        # Process single query
    python main.py -v                        # Create visualization
    python main.py -c                        # Show configuration

Interactive Commands:
    help                    Show available commands
    tools                   List all available tools
    memory                  Show memory status
    clear                   Clear conversation history
    save                    Save conversation to file
    load <filename>         Load conversation from file
    visualize               Create agent flow diagram
    exit/quit               End session

Available Tool Categories:
    • Calculator Tools      - Mathematical operations
    • File Operations       - File management
    • Web Search           - Information retrieval
    • Text Processing      - Text analysis and manipulation
    • Code Processing      - Code generation and debugging
    • Memory Management    - Conversation and data storage
    • Image Processing     - Image analysis and description
"""
    print(help_text)


def show_configuration() -> None:
    """Show current configuration."""
    try:
        config = load_config_file()
        print("Current Configuration:")
        print("=" * 50)
        for key, value in config.items():
            print(f"{key}: {value}")
        print("=" * 50)
        
    except Exception as e:
        print(f"Error loading configuration: {e}")


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Multi-Modal AI Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '-q', '--query',
        type=str,
        help='Process a single query and exit'
    )
    
    parser.add_argument(
        '-v', '--visualize',
        action='store_true',
        help='Create agent flow diagram'
    )
    
    parser.add_argument(
        '-c', '--config',
        action='store_true',
        help='Show current configuration'
    )
    
    parser.add_argument(
        '-i', '--interactive',
        action='store_true',
        help='Run in interactive mode (default)'
    )
    
    args = parser.parse_args()
    
    # Setup environment
    if not setup_environment():
        sys.exit(1)
    
    # Handle different modes
    if args.query:
        run_single_query(args.query)
    elif args.visualize:
        create_visualization()
    elif args.config:
        show_configuration()
    else:
        # Default to interactive mode
        run_interactive_mode()


if __name__ == "__main__":
    main()
