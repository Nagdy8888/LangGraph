# Multi-Modal AI Agent

A comprehensive multi-modal AI agent built with LangGraph that provides various capabilities including mathematical operations, file management, web search, text processing, code generation, memory management, and image processing.

## Features

### 🧮 Calculator Tools
- Basic arithmetic operations (add, subtract, multiply, divide)
- Advanced mathematical functions (power, square root)
- Expression evaluation
- Safe mathematical computation

### 📁 File Operations
- Read, write, create, and delete files
- File listing and information retrieval
- File validation and safety checks
- Support for multiple file types (.txt, .py, .json, .csv, .md, .log)

### 🔍 Web Search
- Information retrieval from web sources
- Configurable search results
- Query processing and result formatting

### 📝 Text Processing
- Text analysis and statistics
- Text summarization
- Language translation (mock implementation)
- Keyword extraction
- Text cleaning and normalization

### 💻 Code Processing
- Code generation based on descriptions
- Code debugging and analysis
- Code explanation and documentation
- Support for multiple programming languages

### 🧠 Memory Management
- Conversation history tracking
- Persistent memory storage
- Memory search and retrieval
- Memory cleanup and optimization

### 🖼️ Image Processing
- Image description and analysis
- Image metadata extraction
- Visual content understanding (mock implementation)

## Installation

### Prerequisites
- Python 3.8+
- OpenAI API key
- Required Python packages (see requirements.txt)

### Setup
1. Clone or download the project
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   ```bash
   # Create a .env file with your OpenAI API key
   echo "OPENAI_API_KEY=your_api_key_here" > .env
   ```
4. Run the agent:
   ```bash
   python main.py
   ```

## Usage

### Interactive Mode (Default)
```bash
python main.py
```

### Single Query Mode
```bash
python main.py -q "What is 2+2?"
```

### Visualization Mode
```bash
python main.py -v
```

### Configuration Mode
```bash
python main.py -c
```

### Help
```bash
python main.py -h
```

## Interactive Commands

When running in interactive mode, you can use these commands:

- `help` - Show available commands and tools
- `tools` - List all available tools by category
- `memory` - Show memory status and statistics
- `clear` - Clear conversation history
- `save` - Save conversation to file
- `load <filename>` - Load conversation from file
- `visualize` - Create agent flow diagram
- `exit`/`quit` - End session

## Tool Categories

### Calculator Tools
- `add(a, b)` - Add two numbers
- `subtract(a, b)` - Subtract two numbers
- `multiply(a, b)` - Multiply two numbers
- `divide(a, b)` - Divide two numbers
- `power(base, exponent)` - Raise to power
- `square_root(number)` - Calculate square root
- `calculate_expression(expression)` - Evaluate mathematical expression

### File Operations
- `read_file(filename)` - Read file contents
- `write_file(filename, content)` - Write content to file
- `create_file(filename, content)` - Create new file
- `delete_file(filename)` - Delete file
- `list_files(directory)` - List files in directory

### Web Search
- `search_web(query, max_results)` - Search the web for information

### Text Processing
- `analyze_text(text)` - Analyze text and provide statistics
- `summarize_text(text, max_length)` - Summarize text
- `translate_text(text, target_language)` - Translate text

### Code Processing
- `generate_code(description, language)` - Generate code
- `debug_code(code, language)` - Debug and analyze code
- `explain_code(code, language)` - Explain code functionality

### Memory Management
- `save_memory(key, value)` - Save information to memory
- `load_memory(key)` - Load information from memory
- `list_memory()` - List all memory keys
- `clear_memory()` - Clear all memory

### Image Processing
- `describe_image(image_path)` - Describe image contents
- `analyze_image(image_path)` - Analyze image metadata

## Configuration

The agent can be configured through the `config.py` file or by creating a custom configuration file. Key configuration options include:

- Model settings (OpenAI model, temperature)
- File size limits
- Memory limits
- Tool categories
- UI messages and prompts

## Architecture

The agent is built using LangGraph and follows a modular architecture:

```
multi_modal_agent.py    # Core agent logic
tools.py               # All available tools
config.py              # Configuration constants
agent_visualizer.py    # Visualization capabilities
utils.py               # Utility functions
main.py                # Entry point
```

## Examples

### Mathematical Operations
```
User: What is 15 * 23 + 45?
Agent: I'll calculate that for you.
[Executing Tool: calculate_expression]
15 * 23 + 45 = 390
```

### File Operations
```
User: Create a file called "notes.txt" with some content
Agent: I'll create that file for you.
[Executing Tool: create_file]
File created successfully: notes.txt
```

### Text Analysis
```
User: Analyze this text: "The quick brown fox jumps over the lazy dog"
Agent: I'll analyze that text for you.
[Executing Tool: analyze_text]
Text Analysis Results:
• Character count: 43
• Word count: 9
• Sentence count: 1
• Average words per sentence: 9.0
```

### Code Generation
```
User: Generate Python code to sort a list
Agent: I'll generate Python code for sorting a list.
[Executing Tool: generate_code]
Generated Python code:
def sort_list(items):
    return sorted(items)
```

## Error Handling

The agent includes comprehensive error handling for:
- Invalid tool calls
- File operation errors
- Network connectivity issues
- Memory management errors
- Input validation errors

## Logging

The agent supports logging with different levels:
- INFO: General information
- WARNING: Warning messages
- ERROR: Error conditions
- DEBUG: Debug information

Logs are saved to the `logs/` directory by default.

## Visualization

The agent can create visual representations of:
- Agent flow diagrams
- Tool usage statistics
- Conversation timelines
- Process flows

Use the `visualize` command or run with `-v` flag to create visualizations.

## Contributing

To contribute to this project:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Check the documentation
- Review the examples
- Create an issue in the repository
- Contact the development team

## Changelog

### Version 1.0.0
- Initial release
- Multi-modal agent with comprehensive tool set
- Interactive and batch processing modes
- Visualization capabilities
- Memory management
- Error handling and logging
