#!/usr/bin/env python3
"""
HCCL API Documentation Generator

This script concatenates all HCCL API documentation files from 
cann-hccl/docs/hccl_customized_dev/ and creates a clean, unified 
API reference document.

The script:
1. Reads all .md files from the API documentation directory
2. Removes HTML tags and irrelevant formatting
3. Extracts function prototypes, descriptions, and parameters
4. Generates a clean, consolidated HCCL_API_ALL.md file
"""

import os
import re
import glob
from pathlib import Path


def clean_html_tags(text):
    """Remove HTML tags and entities from text."""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove HTML entities
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'&\w+;', '', text)  # Remove other HTML entities
    # Remove markdown anchor links
    text = re.sub(r'<a name="[^"]*"></a>', '', text)
    return text.strip()


def extract_table_from_html(content, start_marker):
    """Extract table data from HTML table in the content."""
    parameters = []
    lines = content.split('\n')
    
    in_table = False
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Look for table start - support regex patterns
        if (re.search(start_marker, line) or 
            ('table' in line and '<table' in line) or
            '**表' in line):
            in_table = True
            continue
        
        # Look for table end
        if in_table and ('</table>' in line or '</tbody>' in line):
            break
            
        # Extract table rows
        if in_table and '<td' in line:
            # Find all td elements in surrounding lines
            row_data = []
            j = i
            # Look at current and next few lines to collect full row data
            while j < len(lines) and j < i + 10:  # Extend search range
                if '<td' in lines[j]:
                    # Extract content between <td> and </td>
                    td_content = lines[j]
                    if '<p' in td_content:
                        # Extract content from <p> tags
                        p_match = re.search(r'<p[^>]*>(.*?)</p>', td_content)
                        if p_match:
                            content_text = clean_html_tags(p_match.group(1))
                            if content_text:  # Only add non-empty content
                                row_data.append(content_text)
                
                # Stop at end of table row
                if '</tr>' in lines[j]:
                    break
                j += 1
            
            # For OpParam-style tables, we might have 2 columns (name, description)
            # or 3 columns (name, direction, description)
            if len(row_data) >= 2:
                if len(row_data) == 2:
                    # 2-column format: name, description
                    parameters.append({
                        'name': row_data[0],
                        'direction': '',  # No direction column
                        'description': row_data[1]
                    })
                else:
                    # 3+ column format: name, direction, description
                    parameters.append({
                        'name': row_data[0],
                        'direction': row_data[1] if len(row_data) > 1 else '',
                        'description': row_data[2] if len(row_data) > 2 else row_data[1]
                    })
    
    return parameters


def extract_function_info(content, filename):
    """Extract meaningful information from a single API doc file."""
    lines = content.split('\n')
    
    # Initialize sections
    description = ""
    prototype = ""
    parameters = []
    return_value = ""
    
    # Extract function name from first line
    func_name = filename.replace('.md', '').replace('-', '_')
    for line in lines[:5]:  # Check first few lines
        if line.startswith('# ') and '<a name=' in line:
            func_name = line.split('#')[1].split('<')[0].strip()
            break
    
    # Find sections by Chinese/English markers
    current_section = None
    in_code_block = False
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
            
        # Handle code blocks
        if line.startswith('```'):
            in_code_block = not in_code_block
            if current_section == 'prototype' and not in_code_block:
                current_section = None
            continue
            
        # Identify sections by Chinese headers
        if ('## 功能说明' in line or '## Function Description' in line or 
            '## 简介' in line or '## Description' in line):
            current_section = 'description'
            continue
        elif ('## 函数原型' in line or '## Function Prototype' in line or 
              '## 原型定义' in line or '## Prototype' in line):
            current_section = 'prototype'
            continue
        elif ('## 参数说明' in line or '## Parameters' in line or 
              '## 成员介绍' in line or '## Members' in line or
              '## 成员说明' in line):
            current_section = 'parameters'
            # Extract parameters from HTML table
            remaining_content = '\n'.join(lines[i:])
            parameters = extract_table_from_html(remaining_content, '参数说明|成员介绍|成员说明')
            current_section = None
            continue
        elif ('## 返回值' in line or '## Return Value' in line or
              '## 返回值说明' in line):
            current_section = 'return'
            continue
        elif ('## 约束说明' in line or '## 约束' in line or '## Constraints' in line):
            current_section = None
            continue
            
        # Collect content based on current section
        if current_section == 'description' and line and not line.startswith('##'):
            if not description:
                description = clean_html_tags(line)
        elif current_section == 'prototype' and in_code_block and line:
            prototype += line + '\n'
        elif current_section == 'return' and line and not line.startswith('##'):
            if not return_value:
                return_value = clean_html_tags(line)
    
    return {
        'filename': filename,
        'function_name': func_name,
        'description': description,
        'prototype': prototype.strip(),
        'parameters': parameters,
        'return_value': return_value
    }


def should_include_file(filename):
    """Determine if a file should be included in the API documentation."""
    # Skip certain files that are not individual API functions
    skip_patterns = [
        'README.md',
        '简介',
        '概述',
        '总体流程',
        '数据类型说明',
        '算法开发',
        '背景知识',
        '如何验证',
        '增加',
        '适配',
        '通信',
        'figures'
    ]
    
    for pattern in skip_patterns:
        if pattern in filename:
            return False
    return filename.endswith('.md')


def generate_api_documentation(docs_dir, output_file):
    """Generate consolidated API documentation."""
    
    # Find all markdown files
    md_files = []
    for file_path in glob.glob(os.path.join(docs_dir, "*.md")):
        filename = os.path.basename(file_path)
        if should_include_file(filename):
            md_files.append(file_path)
    
    # Sort files alphabetically for consistent output
    md_files.sort()
    
    print(f"Processing {len(md_files)} API documentation files...")
    
    # Process each file
    api_functions = []
    for file_path in md_files:
        filename = os.path.basename(file_path)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            func_info = extract_function_info(content, filename)
            if func_info['prototype'] or func_info['description']:
                api_functions.append(func_info)
                print(f"✓ Processed: {filename}")
            else:
                print(f"⚠ Skipped (no meaningful content): {filename}")
                
        except Exception as e:
            print(f"✗ Error processing {filename}: {e}")
    
    # Generate output
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# HCCL API Complete Reference\n\n")
        f.write("This document provides a comprehensive reference for all HCCL APIs.\n")
        f.write("Generated automatically from the official HCCL documentation.\n\n")
        f.write("---\n\n")
        
        for func in api_functions:
            # Use extracted function name
            func_name = func['function_name']
            
            f.write(f"## {func_name}\n\n")
            
            if func['description']:
                f.write(f"**Description:** {func['description']}\n\n")
            
            if func['prototype']:
                f.write("**Prototype:**\n```cpp\n")
                f.write(func['prototype'])
                f.write("\n```\n\n")
            
            if func['parameters']:
                f.write("**Parameters:**\n\n")
                f.write("| Parameter | Direction | Description |\n")
                f.write("|-----------|-----------|-------------|\n")
                for param in func['parameters']:
                    f.write(f"| {param['name']} | {param['direction']} | {param['description']} |\n")
                f.write("\n")
            
            if func['return_value']:
                f.write(f"**Return Value:** {func['return_value']}\n\n")
            
            f.write("---\n\n")
    
    print(f"\n✅ Generated consolidated API documentation: {output_file}")
    print(f"📊 Total functions documented: {len(api_functions)}")


def main():
    # Set up paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    docs_dir = project_root / "cann-hccl" / "docs" / "hccl_customized_dev"
    output_file = project_root / "HCCL_API_ALL.md"
    
    if not docs_dir.exists():
        print(f"Error: Documentation directory not found: {docs_dir}")
        return 1
    
    print(f"📁 Source directory: {docs_dir}")
    print(f"📄 Output file: {output_file}")
    print()
    
    try:
        generate_api_documentation(str(docs_dir), str(output_file))
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())