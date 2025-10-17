"""
Multi-Modal Agent Tools Module.

This module contains all the tools available to the multi-modal agent,
including calculator, file operations, web search, text processing, and more.
"""

import os
import json
import math
import requests
import base64
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from pathlib import Path

from langchain_core.tools import tool
from config import (
    MAX_FILE_SIZE, MAX_SEARCH_RESULTS, MAX_MEMORY_ENTRIES,
    SUPPORTED_FILE_TYPES, MEMORY_FILE, CONVERSATION_FILE,
    ERROR_MESSAGES, SUCCESS_MESSAGES, WORKSPACE_DIR
)


# ==================== CALCULATOR TOOLS ====================

@tool
def add(a: float, b: float) -> str:
    """Add two numbers together.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        Sum of the two numbers
    """
    result = a + b
    return f"{a} + {b} = {result}"


@tool
def subtract(a: float, b: float) -> str:
    """Subtract second number from first number.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        Difference of the two numbers
    """
    result = a - b
    return f"{a} - {b} = {result}"


@tool
def multiply(a: float, b: float) -> str:
    """Multiply two numbers.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        Product of the two numbers
    """
    result = a * b
    return f"{a} × {b} = {result}"


@tool
def divide(a: float, b: float) -> str:
    """Divide first number by second number.
    
    Args:
        a: Dividend
        b: Divisor
        
    Returns:
        Quotient of the division
    """
    if b == 0:
        return "Error: Division by zero is not allowed"
    result = a / b
    return f"{a} ÷ {b} = {result}"


@tool
def power(base: float, exponent: float) -> str:
    """Raise base to the power of exponent.
    
    Args:
        base: Base number
        exponent: Exponent
        
    Returns:
        Result of the power operation
    """
    result = base ** exponent
    return f"{base}^{exponent} = {result}"


@tool
def square_root(number: float) -> str:
    """Calculate square root of a number.
    
    Args:
        number: Number to find square root of
        
    Returns:
        Square root of the number
    """
    if number < 0:
        return "Error: Cannot calculate square root of negative number"
    result = math.sqrt(number)
    return f"√{number} = {result}"


@tool
def calculate_expression(expression: str) -> str:
    """Safely evaluate a mathematical expression.
    
    Args:
        expression: Mathematical expression as string
        
    Returns:
        Result of the expression evaluation
    """
    try:
        # Only allow safe mathematical operations
        allowed_chars = set('0123456789+-*/().,e ')
        if not all(c in allowed_chars for c in expression):
            return "Error: Expression contains invalid characters"
        
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"


# ==================== FILE OPERATION TOOLS ====================

@tool
def read_file(filename: str) -> str:
    """Read contents of a text file.
    
    Args:
        filename: Name of the file to read
        
    Returns:
        File contents or error message
    """
    try:
        file_path = Path(WORKSPACE_DIR) / filename
        
        if not file_path.exists():
            return ERROR_MESSAGES['file_not_found'].format(filename=filename)
        
        if file_path.stat().st_size > MAX_FILE_SIZE:
            return ERROR_MESSAGES['file_too_large'].format(
                filename=filename, max_size=MAX_FILE_SIZE // (1024 * 1024)
            )
        
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        return f"File contents of '{filename}':\n{content}"
    
    except PermissionError:
        return ERROR_MESSAGES['permission_denied'].format(filename=filename)
    except Exception as e:
        return f"Error reading file: {str(e)}"


@tool
def write_file(filename: str, content: str) -> str:
    """Write content to a text file.
    
    Args:
        filename: Name of the file to write
        content: Content to write to the file
        
    Returns:
        Success or error message
    """
    try:
        file_path = Path(WORKSPACE_DIR) / filename
        
        # Create directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)
        
        return SUCCESS_MESSAGES['file_saved'].format(filename=filename)
    
    except Exception as e:
        return f"Error writing file: {str(e)}"


@tool
def create_file(filename: str, content: str = "") -> str:
    """Create a new file with optional content.
    
    Args:
        filename: Name of the file to create
        content: Optional initial content
        
    Returns:
        Success or error message
    """
    try:
        file_path = Path(WORKSPACE_DIR) / filename
        
        if file_path.exists():
            return f"File '{filename}' already exists"
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)
        
        return SUCCESS_MESSAGES['file_created'].format(filename=filename)
    
    except Exception as e:
        return f"Error creating file: {str(e)}"


@tool
def delete_file(filename: str) -> str:
    """Delete a file.
    
    Args:
        filename: Name of the file to delete
        
    Returns:
        Success or error message
    """
    try:
        file_path = Path(WORKSPACE_DIR) / filename
        
        if not file_path.exists():
            return ERROR_MESSAGES['file_not_found'].format(filename=filename)
        
        file_path.unlink()
        return SUCCESS_MESSAGES['file_deleted'].format(filename=filename)
    
    except PermissionError:
        return ERROR_MESSAGES['permission_denied'].format(filename=filename)
    except Exception as e:
        return f"Error deleting file: {str(e)}"


@tool
def list_files(directory: str = ".") -> str:
    """List files in a directory.
    
    Args:
        directory: Directory to list files from (default: current directory)
        
    Returns:
        List of files in the directory
    """
    try:
        dir_path = Path(WORKSPACE_DIR) / directory
        
        if not dir_path.exists():
            return f"Directory '{directory}' not found"
        
        if not dir_path.is_dir():
            return f"'{directory}' is not a directory"
        
        files = []
        for item in dir_path.iterdir():
            if item.is_file():
                size = item.stat().st_size
                files.append(f"📄 {item.name} ({size} bytes)")
            elif item.is_dir():
                files.append(f"📁 {item.name}/")
        
        if not files:
            return f"No files found in '{directory}'"
        
        return f"Files in '{directory}':\n" + "\n".join(files)
    
    except Exception as e:
        return f"Error listing files: {str(e)}"


# ==================== WEB SEARCH TOOLS ====================

@tool
def search_web(query: str, max_results: int = MAX_SEARCH_RESULTS) -> str:
    """Search the web for information.
    
    Args:
        query: Search query
        max_results: Maximum number of results to return
        
    Returns:
        Search results or error message
    """
    try:
        # This is a mock implementation - in a real scenario, you'd use a search API
        # For now, we'll simulate search results
        results = [
            f"Search result {i+1} for '{query}': This is a simulated search result that would contain relevant information about your query.",
            f"Search result {i+1} for '{query}': Another simulated result with additional context and details.",
            f"Search result {i+1} for '{query}': A third result providing different perspectives on the topic."
        ]
        
        limited_results = results[:max_results]
        return f"Web search results for '{query}':\n\n" + "\n\n".join(limited_results)
    
    except Exception as e:
        return f"Error searching web: {str(e)}"


# ==================== TEXT PROCESSING TOOLS ====================

@tool
def analyze_text(text: str) -> str:
    """Analyze text and provide statistics and insights.
    
    Args:
        text: Text to analyze
        
    Returns:
        Text analysis results
    """
    try:
        words = text.split()
        sentences = text.split('.')
        paragraphs = text.split('\n\n')
        
        analysis = {
            "Character count": len(text),
            "Word count": len(words),
            "Sentence count": len([s for s in sentences if s.strip()]),
            "Paragraph count": len([p for p in paragraphs if p.strip()]),
            "Average words per sentence": round(len(words) / max(len([s for s in sentences if s.strip()]), 1), 2),
            "Most common words": "N/A"  # Could implement word frequency analysis
        }
        
        result = "Text Analysis Results:\n"
        for key, value in analysis.items():
            result += f"• {key}: {value}\n"
        
        return result
    
    except Exception as e:
        return f"Error analyzing text: {str(e)}"


@tool
def summarize_text(text: str, max_length: int = 200) -> str:
    """Summarize text to a specified length.
    
    Args:
        text: Text to summarize
        max_length: Maximum length of summary
        
    Returns:
        Text summary
    """
    try:
        if len(text) <= max_length:
            return f"Text is already short enough:\n{text}"
        
        # Simple summarization - take first part and last part
        words = text.split()
        if len(words) <= max_length // 5:  # If very short, return as is
            return text
        
        # Take first and last portions
        first_part = " ".join(words[:max_length // 10])
        last_part = " ".join(words[-max_length // 10:])
        
        summary = f"{first_part}... [truncated] ...{last_part}"
        return f"Summary (max {max_length} chars):\n{summary}"
    
    except Exception as e:
        return f"Error summarizing text: {str(e)}"


@tool
def translate_text(text: str, target_language: str = "Spanish") -> str:
    """Translate text to a target language.
    
    Args:
        text: Text to translate
        target_language: Target language for translation
        
    Returns:
        Translated text (mock implementation)
    """
    try:
        # This is a mock translation - in a real scenario, you'd use a translation API
        mock_translations = {
            "Spanish": f"[ES] {text}",
            "French": f"[FR] {text}",
            "German": f"[DE] {text}",
            "Italian": f"[IT] {text}",
            "Portuguese": f"[PT] {text}"
        }
        
        translation = mock_translations.get(target_language, f"[{target_language}] {text}")
        return f"Translation to {target_language}:\n{translation}"
    
    except Exception as e:
        return f"Error translating text: {str(e)}"


# ==================== CODE PROCESSING TOOLS ====================

@tool
def generate_code(description: str, language: str = "python") -> str:
    """Generate code based on a description.
    
    Args:
        description: Description of what the code should do
        language: Programming language (default: python)
        
    Returns:
        Generated code
    """
    try:
        # This is a mock implementation - in a real scenario, you'd use an AI model
        mock_code = f"""
# Generated {language} code for: {description}

def main():
    # TODO: Implement the functionality described
    # {description}
    pass

if __name__ == "__main__":
    main()
"""
        return f"Generated {language} code:\n{mock_code}"
    
    except Exception as e:
        return f"Error generating code: {str(e)}"


@tool
def debug_code(code: str, language: str = "python") -> str:
    """Analyze code for potential issues.
    
    Args:
        code: Code to debug
        language: Programming language
        
    Returns:
        Debug analysis results
    """
    try:
        # Basic syntax checking (mock implementation)
        issues = []
        
        if "print(" in code and ")" not in code:
            issues.append("Potential syntax error: Unclosed print statement")
        
        if "def " in code and ":" not in code:
            issues.append("Potential syntax error: Function definition missing colon")
        
        if "if " in code and ":" not in code:
            issues.append("Potential syntax error: If statement missing colon")
        
        if not issues:
            return f"Code analysis for {language}:\n✅ No obvious syntax issues found"
        
        result = f"Code analysis for {language}:\n"
        for i, issue in enumerate(issues, 1):
            result += f"⚠️  {i}. {issue}\n"
        
        return result
    
    except Exception as e:
        return f"Error debugging code: {str(e)}"


@tool
def explain_code(code: str, language: str = "python") -> str:
    """Explain what code does.
    
    Args:
        code: Code to explain
        language: Programming language
        
    Returns:
        Code explanation
    """
    try:
        # Mock explanation - in a real scenario, you'd use an AI model
        explanation = f"""
Code Explanation for {language}:

This code appears to be written in {language}. Here's what it does:

1. The code structure suggests it's a {language} program
2. It contains various programming constructs
3. The functionality depends on the specific implementation

For a detailed explanation, please provide more context about what this code is supposed to accomplish.
"""
        return explanation
    
    except Exception as e:
        return f"Error explaining code: {str(e)}"


# ==================== MEMORY MANAGEMENT TOOLS ====================

@tool
def save_memory(key: str, value: str) -> str:
    """Save information to agent memory.
    
    Args:
        key: Memory key/identifier
        value: Information to store
        
    Returns:
        Success or error message
    """
    try:
        memory_file = Path(WORKSPACE_DIR) / MEMORY_FILE
        
        # Load existing memory
        memory = {}
        if memory_file.exists():
            with open(memory_file, 'r', encoding='utf-8') as f:
                memory = json.load(f)
        
        # Add new entry
        memory[key] = {
            "value": value,
            "timestamp": datetime.now().isoformat()
        }
        
        # Limit memory size
        if len(memory) > MAX_MEMORY_ENTRIES:
            # Remove oldest entries
            sorted_items = sorted(memory.items(), key=lambda x: x[1]["timestamp"])
            for old_key, _ in sorted_items[:len(memory) - MAX_MEMORY_ENTRIES]:
                del memory[old_key]
        
        # Save memory
        with open(memory_file, 'w', encoding='utf-8') as f:
            json.dump(memory, f, indent=2)
        
        return SUCCESS_MESSAGES['memory_saved']
    
    except Exception as e:
        return f"Error saving memory: {str(e)}"


@tool
def load_memory(key: str) -> str:
    """Load information from agent memory.
    
    Args:
        key: Memory key/identifier
        
    Returns:
        Stored information or error message
    """
    try:
        memory_file = Path(WORKSPACE_DIR) / MEMORY_FILE
        
        if not memory_file.exists():
            return "No memory file found"
        
        with open(memory_file, 'r', encoding='utf-8') as f:
            memory = json.load(f)
        
        if key not in memory:
            return f"Memory key '{key}' not found"
        
        entry = memory[key]
        return f"Memory '{key}': {entry['value']} (saved: {entry['timestamp']})"
    
    except Exception as e:
        return f"Error loading memory: {str(e)}"


@tool
def list_memory() -> str:
    """List all memory keys.
    
    Returns:
        List of memory keys
    """
    try:
        memory_file = Path(WORKSPACE_DIR) / MEMORY_FILE
        
        if not memory_file.exists():
            return "No memory file found"
        
        with open(memory_file, 'r', encoding='utf-8') as f:
            memory = json.load(f)
        
        if not memory:
            return "Memory is empty"
        
        keys = list(memory.keys())
        return f"Memory keys ({len(keys)}):\n" + "\n".join(f"• {key}" for key in keys)
    
    except Exception as e:
        return f"Error listing memory: {str(e)}"


@tool
def clear_memory() -> str:
    """Clear all agent memory.
    
    Returns:
        Success message
    """
    try:
        memory_file = Path(WORKSPACE_DIR) / MEMORY_FILE
        
        if memory_file.exists():
            memory_file.unlink()
        
        return SUCCESS_MESSAGES['memory_cleared']
    
    except Exception as e:
        return f"Error clearing memory: {str(e)}"


# ==================== IMAGE PROCESSING TOOLS ====================

@tool
def describe_image(image_path: str) -> str:
    """Describe an image file.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Image description
    """
    try:
        file_path = Path(WORKSPACE_DIR) / image_path
        
        if not file_path.exists():
            return ERROR_MESSAGES['file_not_found'].format(filename=image_path)
        
        # Mock image description - in a real scenario, you'd use a vision model
        return f"Image description for '{image_path}': This is a mock description. In a real implementation, this would use a vision model to analyze the image and provide a detailed description of its contents, objects, colors, and composition."
    
    except Exception as e:
        return f"Error describing image: {str(e)}"


@tool
def analyze_image(image_path: str) -> str:
    """Analyze an image and provide detailed information.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Image analysis results
    """
    try:
        file_path = Path(WORKSPACE_DIR) / image_path
        
        if not file_path.exists():
            return ERROR_MESSAGES['file_not_found'].format(filename=image_path)
        
        # Mock image analysis
        analysis = {
            "File size": f"{file_path.stat().st_size} bytes",
            "File type": file_path.suffix,
            "Analysis": "Mock analysis - would include object detection, color analysis, text recognition, etc."
        }
        
        result = f"Image Analysis for '{image_path}':\n"
        for key, value in analysis.items():
            result += f"• {key}: {value}\n"
        
        return result
    
    except Exception as e:
        return f"Error analyzing image: {str(e)}"


# ==================== TOOL REGISTRY ====================

# All available tools
ALL_TOOLS = [
    # Calculator tools
    add, subtract, multiply, divide, power, square_root, calculate_expression,
    
    # File operation tools
    read_file, write_file, create_file, delete_file, list_files,
    
    # Web search tools
    search_web,
    
    # Text processing tools
    analyze_text, summarize_text, translate_text,
    
    # Code processing tools
    generate_code, debug_code, explain_code,
    
    # Memory management tools
    save_memory, load_memory, list_memory, clear_memory,
    
    # Image processing tools
    describe_image, analyze_image
]

# Tool categories for organization
TOOL_CATEGORIES = {
    "Calculator": [add, subtract, multiply, divide, power, square_root, calculate_expression],
    "File Operations": [read_file, write_file, create_file, delete_file, list_files],
    "Web Search": [search_web],
    "Text Processing": [analyze_text, summarize_text, translate_text],
    "Code Processing": [generate_code, debug_code, explain_code],
    "Memory Management": [save_memory, load_memory, list_memory, clear_memory],
    "Image Processing": [describe_image, analyze_image]
}
