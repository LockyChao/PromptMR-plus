#!/usr/bin/env python3
"""
Folder Structure Generator

This script generates a visual representation of a folder structure,
similar to the tree command but with custom formatting that matches
the example provided.

Usage:
    python generate_folder_structure.py /path/to/root/folder
"""

import os
import sys
from pathlib import Path


def generate_folder_structure(root_path, output_file=None):
    """
    Generate a folder structure visualization for the given root path.
    
    Args:
        root_path (str): The root directory to analyze
        output_file (str, optional): Output file path. If None, prints to stdout.
    """
    
    def get_tree_structure(directory, prefix="", is_last=True, current_depth=0):
        """
        Recursively build the tree structure.
        
        Args:
            directory (Path): Current directory path
            prefix (str): Current prefix for indentation
            is_last (bool): Whether this is the last item in the current level
            current_depth (int): Current depth level
        """
        lines = []
        
        # Get directory name
        if current_depth == 0:
            lines.append(str(directory.name))
        
        try:
            # Get all items in the directory
            items = sorted(directory.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
            
            for i, item in enumerate(items):
                is_last_item = i == len(items) - 1
                
                # Determine the connector
                if current_depth == 0:
                    connector = "- "
                else:
                    connector = "- " if is_last_item else "- "
                
                # Create the current line
                indent = "-" * (current_depth + 1) + " "
                current_line = indent + item.name
                
                if item.is_dir():
                    lines.append(current_line)
                    # Recursively process subdirectories
                    sub_lines = get_tree_structure(
                        item, 
                        prefix + ("  " if is_last else "│ "),
                        is_last_item,
                        current_depth + 1
                    )
                    lines.extend(sub_lines)
                else:
                    lines.append(current_line)
                    
        except PermissionError:
            lines.append(prefix + "├── [Permission Denied]")
            
        return lines
    
    # Convert to Path object
    root = Path(root_path)
    
    if not root.exists():
        print(f"Error: Path '{root_path}' does not exist.")
        return
    
    if not root.is_dir():
        print(f"Error: Path '{root_path}' is not a directory.")
        return
    
    print(f"Generating folder structure for: {root.absolute()}")
    print("=" * 50)
    
    # Generate the tree structure
    tree_lines = get_tree_structure(root)
    
    # Output the result
    output_text = "\n".join(tree_lines)
    
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(output_text)
            print(f"Folder structure saved to: {output_file}")
        except Exception as e:
            print(f"Error writing to file: {e}")
            print("\nFolder structure:")
            print(output_text)
    else:
        print("\nFolder structure:")
        print(output_text)


def generate_folder_structure_simple(root_path, output_file=None):
    """
    Generate a simpler folder structure that matches your exact example format.
    
    Args:
        root_path (str): The root directory to analyze
        output_file (str, optional): Output file path. If None, prints to stdout.
    """
    
    def walk_directory(directory, depth=0):
        """Walk through directory and yield formatted lines."""
        lines = []
        
        # Create prefix based on depth
        prefix = "-" * depth + " " if depth > 0 else ""
        
        try:
            # Get all items in the directory, sorted
            items = sorted(directory.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
            
            for item in items:
                current_line = prefix + item.name
                lines.append(current_line)
                
                # If it's a directory, recursively process it
                if item.is_dir():
                    sub_lines = walk_directory(item, depth + 1)
                    lines.extend(sub_lines)
                    
        except PermissionError:
            lines.append(prefix + "[Permission Denied]")
            
        return lines
    
    # Convert to Path object
    root = Path(root_path)
    
    if not root.exists():
        print(f"Error: Path '{root_path}' does not exist.")
        return
    
    if not root.is_dir():
        print(f"Error: Path '{root_path}' is not a directory.")
        return
    
    print(f"Generating folder structure for: {root.absolute()}")
    print("=" * 50)
    
    # Start with the root directory name
    lines = [root.name]
    
    # Add subdirectories and files
    sub_lines = walk_directory(root, 1)
    lines.extend(sub_lines)
    
    # Join all lines
    output_text = "\n".join(lines)
    
    # Output the result
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(output_text)
            print(f"Folder structure saved to: {output_file}")
        except Exception as e:
            print(f"Error writing to file: {e}")
            print("\nFolder structure:")
            print(output_text)
    else:
        print("\nFolder structure:")
        print(output_text)


def main():
    """Main function to handle command line arguments."""
    if len(sys.argv) < 2:
        print("Usage: python generate_folder_structure.py <root_folder> [output_file]")
        print("Example: python generate_folder_structure.py /path/to/folder structure.txt")
        sys.exit(1)
    
    root_folder = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Use the simple format that matches your example
    generate_folder_structure_simple(root_folder, output_file)


if __name__ == "__main__":
    main()