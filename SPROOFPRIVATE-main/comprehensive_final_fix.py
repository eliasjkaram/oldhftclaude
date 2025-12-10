#\!/usr/bin/env python3
"""Final comprehensive fix for all remaining syntax errors"""

import re
import os

def fix_all_files():
    files_to_fix = [
        "src/misc/enhanced_ultimate_engine.py",
        "src/misc/enhanced_trading_gui.py", 
        "src/misc/ULTIMATE_COMPLEX_TRADING_GUI.py",
        "src/misc/FINAL_ULTIMATE_COMPLETE_SYSTEM.py"
    ]
    
    for file_path in files_to_fix:
        if os.path.exists(file_path):
            print(f"Fixing {file_path}...")
            
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Fix missing closing parentheses
            lines = content.split('\n')
            fixed_lines = []
            
            for i, line in enumerate(lines):
                # Check for common missing closing parentheses patterns
                if any(pattern in line for pattern in ['.json(', '.get(', '.upper(', '.lower(', 
                                                       '.strip(', '.split(', '.start(', '.pack(',
                                                       '.grid(', '.place(', '.forget(', '.update(',
                                                       '.mean(', '.std(', '.sum(', '.min(', '.max(']):
                    # Count parentheses
                    open_count = line.count('(')
                    close_count = line.count(')')
                    if open_count > close_count:
                        # Add missing closing parentheses
                        line = line + ')' * (open_count - close_count)
                
                # Fix dict initialization patterns
                if line.strip().endswith('= {}'):
                    next_line_idx = i + 1
                    if next_line_idx < len(lines) and lines[next_line_idx].strip() and not lines[next_line_idx].strip().startswith('}'):
                        if lines[next_line_idx].strip()[0] in ["'", '"', '(']:
                            line = line.replace('= {}', '= {')
                
                # Fix list initialization patterns  
                if line.strip().endswith('= []'):
                    next_line_idx = i + 1
                    if next_line_idx < len(lines) and lines[next_line_idx].strip() and not lines[next_line_idx].strip().startswith(']'):
                        if lines[next_line_idx].strip()[0] in ["'", '"', '(']:
                            line = line.replace('= []', '= [')
                            
                fixed_lines.append(line)
            
            content = '\n'.join(fixed_lines)
            
            with open(file_path, 'w') as f:
                f.write(content)
            
            print(f"  Fixed {file_path}")

if __name__ == "__main__":
    fix_all_files()
    print("Done fixing all files\!")
