"""
Configuration constants for the Multi-Modal Agent system.

This module contains all configuration constants used across the multi-modal agent components.
"""

# Model Configuration
DEFAULT_MODEL = "gpt-4o"
DEFAULT_TEMPERATURE = 0.1
EMBEDDING_MODEL = "text-embedding-3-small"

# File and Directory Paths
WORKSPACE_DIR = r"C:\Nagdy\Mustafa\DataCamp\LangGraph\13 Ai agent 6"
DATA_DIR = "data"
LOGS_DIR = "logs"
OUTPUT_DIR = "output"

# UI Messages
WELCOME_MESSAGE = """
=== MULTI-MODAL AI AGENT ===
I'm your intelligent assistant with multiple capabilities:
• Mathematical calculations and data analysis
• File operations (read, write, create, delete)
• Web search and information retrieval
• Text processing and document management
• Code generation and debugging
• Image analysis and description
• Memory management and conversation history

Type 'help' for available commands or 'exit' to quit.
"""
EXIT_COMMANDS = ['exit', 'quit', 'bye']
QUESTION_PROMPT = "\n🤖 What can I help you with? "
ANSWER_HEADER = "\n=== RESPONSE ==="
TOOL_CALL_PREFIX = "🔧 Executing Tool:"
TOOL_RESULT_PREFIX = "✅ Tool Result:"
TOOL_EXECUTION_COMPLETE = "🔄 Tool execution complete. Processing response..."
TOOL_NOT_FOUND_ERROR = "❌ Tool not found. Please check available tools."

# Help Messages
HELP_MESSAGE = """
Available Commands:
• 'help' - Show this help message
• 'tools' - List all available tools
• 'memory' - Show conversation memory
• 'clear' - Clear conversation history
• 'save' - Save conversation to file
• 'load <filename>' - Load conversation from file
• 'visualize' - Show agent flow diagram
• 'exit'/'quit' - End session

Available Tools:
• Calculator functions (add, subtract, multiply, divide, power, sqrt)
• File operations (read, write, create, delete, list files)
• Web search (search_web)
• Text processing (analyze_text, summarize, translate)
• Code operations (generate_code, debug_code, explain_code)
• Memory operations (save_memory, load_memory, clear_memory)
• Image operations (describe_image, analyze_image)
"""

# System Prompts
SYSTEM_PROMPT = """
You are a multi-modal AI assistant with access to various tools and capabilities. 
You can help users with:

1. Mathematical calculations and data analysis
2. File operations and document management
3. Web search and information retrieval
4. Text processing and language tasks
5. Code generation and debugging
6. Memory management and conversation tracking
7. Image analysis and description

Always be helpful, accurate, and provide clear explanations. When using tools:
- Explain what you're doing and why
- Show the results clearly
- Ask for clarification if needed
- Provide context and insights

Use the available tools to provide the best possible assistance. If you need to use multiple tools, 
explain your reasoning and execute them in a logical order.
"""

# Tool Configuration
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_SEARCH_RESULTS = 5
MAX_MEMORY_ENTRIES = 100
SUPPORTED_FILE_TYPES = ['.txt', '.py', '.json', '.csv', '.md', '.log']

# Visualization Configuration
GRAPH_WIDTH = 14
GRAPH_HEIGHT = 10
NODE_COLORS = {
    'agent': '#4CAF50',           # Green
    'calculator': '#2196F3',      # Blue
    'file_manager': '#FF9800',    # Orange
    'web_search': '#9C27B0',      # Purple
    'text_processor': '#00BCD4',  # Cyan
    'code_processor': '#795548',  # Brown
    'memory_manager': '#607D8B',  # Blue Grey
    'image_processor': '#E91E63', # Pink
    'start': '#FF5722',           # Deep Orange
    'end': '#F44336'              # Red
}
EDGE_COLORS = {
    'normal': '#666666',          # Gray
    'conditional': '#9C27B0',     # Purple
    'tool_call': '#FF9800'        # Orange
}

# Memory Configuration
MEMORY_FILE = "agent_memory.json"
CONVERSATION_FILE = "conversation_history.txt"
MAX_CONVERSATION_LENGTH = 1000

# Error Messages
ERROR_MESSAGES = {
    'file_not_found': "❌ File not found: {filename}",
    'permission_denied': "❌ Permission denied: {filename}",
    'invalid_file_type': "❌ Unsupported file type: {file_type}",
    'file_too_large': "❌ File too large: {filename} (max {max_size}MB)",
    'network_error': "❌ Network error: {error}",
    'tool_error': "❌ Tool error: {tool_name} - {error}",
    'memory_error': "❌ Memory error: {error}",
    'validation_error': "❌ Validation error: {error}"
}

# Success Messages
SUCCESS_MESSAGES = {
    'file_saved': "✅ File saved successfully: {filename}",
    'file_created': "✅ File created successfully: {filename}",
    'file_deleted': "✅ File deleted successfully: {filename}",
    'memory_saved': "✅ Memory saved successfully",
    'memory_loaded': "✅ Memory loaded successfully",
    'memory_cleared': "✅ Memory cleared successfully",
    'conversation_saved': "✅ Conversation saved successfully",
    'tool_executed': "✅ Tool executed successfully: {tool_name}"
}
