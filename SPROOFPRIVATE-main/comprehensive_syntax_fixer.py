#!/usr/bin/env python3
"""
Comprehensive syntax fixer for all 4 files
"""

import ast
import re
import os
import sys

def find_syntax_errors(filepath):
    """Find syntax errors in a file"""
    errors = []
    try:
        with open(filepath, 'r') as f:
            source = f.read()
        ast.parse(source)
        return []
    except SyntaxError as e:
        return [(e.lineno, e.msg, e.text)]
    except Exception as e:
        return [(0, str(e), "")]

def fix_enhanced_ultimate_engine(filepath):
    """Fix enhanced_ultimate_engine.py specific issues"""
    print(f"\n=== Fixing {filepath} ===")
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    fixed_lines = []
    for i, line in enumerate(lines):
        # Fix extra closing parentheses
        if i == 417 and line.strip().endswith('))'):
            line = line.rstrip() + '\n'
            if line.endswith('))\n'):
                line = line[:-2] + '\n'
        
        # Fix line 381 - wrong parenthesis placement
        if i == 380 and 'norm_cdf_gpu(d1) -)' in line:
            line = line.replace('norm_cdf_gpu(d1) -)', 'norm_cdf_gpu(d1) -')
        
        fixed_lines.append(line)
    
    with open(filepath, 'w') as f:
        f.writelines(fixed_lines)
    
    print(f"✅ Applied fixes to {filepath}")

def fix_enhanced_trading_gui(filepath):
    """Fix enhanced_trading_gui.py specific issues"""
    print(f"\n=== Fixing {filepath} ===")
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    fixed_lines = []
    for i, line in enumerate(lines):
        # Fix missing parentheses
        if i == 1368 and 'data[\'volume_sma\'] = data[\'volume\'].rolling(20).mean(' in line:
            line = line.replace('.mean(', '.mean()')
        
        # Fix return statement indentation
        if i == 1381 and line.strip().startswith('return features'):
            # Make sure it's properly indented
            line = '            return features if len(features) > 0 else None\n'
        
        fixed_lines.append(line)
    
    with open(filepath, 'w') as f:
        f.writelines(fixed_lines)
    
    print(f"✅ Applied fixes to {filepath}")

def fix_ultimate_complex_trading_gui(filepath):
    """Fix ULTIMATE_COMPLEX_TRADING_GUI.py specific issues"""
    print(f"\n=== Fixing {filepath} ===")
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    fixed_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Fix unmatched parentheses around line 1603
        if i >= 1600 and i <= 1605:
            # Check for lone closing parenthesis
            if line.strip() == ')' and i > 0:
                # Check if previous line has an opening that needs closing
                prev_line = fixed_lines[-1] if fixed_lines else ''
                if prev_line.count('(') > prev_line.count(')'):
                    # This is likely a valid closing parenthesis
                    pass
                else:
                    # Skip this line - it's an extra closing parenthesis
                    i += 1
                    continue
        
        fixed_lines.append(line)
        i += 1
    
    with open(filepath, 'w') as f:
        f.writelines(fixed_lines)
    
    print(f"✅ Applied fixes to {filepath}")

def fix_final_ultimate_complete_system(filepath):
    """Fix FINAL_ULTIMATE_COMPLETE_SYSTEM.py specific issues"""
    print(f"\n=== Fixing {filepath} ===")
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    fixed_lines = []
    for i, line in enumerate(lines):
        # Fix indentation issues
        if i >= 1310 and i <= 1315:
            # Ensure proper indentation for method definitions
            if line.strip().startswith('def ') and not line.startswith('    def '):
                line = '    ' + line.lstrip()
        
        # Fix any standalone update_time() calls
        if 'self.update_time(' in line and line.strip().endswith('('):
            line = line.replace('self.update_time(', 'self.update_time()')
        
        fixed_lines.append(line)
    
    with open(filepath, 'w') as f:
        f.writelines(fixed_lines)
    
    print(f"✅ Applied fixes to {filepath}")

def main():
    """Main function"""
    files_to_fix = [
        ('src/misc/enhanced_ultimate_engine.py', fix_enhanced_ultimate_engine),
        ('src/misc/enhanced_trading_gui.py', fix_enhanced_trading_gui),
        ('src/misc/ULTIMATE_COMPLEX_TRADING_GUI.py', fix_ultimate_complex_trading_gui),
        ('src/misc/FINAL_ULTIMATE_COMPLETE_SYSTEM.py', fix_final_ultimate_complete_system)
    ]
    
    print("=== Comprehensive Syntax Fixer ===")
    
    for filepath, fixer_func in files_to_fix:
        if os.path.exists(filepath):
            # Find current errors
            errors = find_syntax_errors(filepath)
            if errors:
                print(f"\nFound {len(errors)} syntax error(s) in {filepath}")
                for lineno, msg, text in errors:
                    print(f"  Line {lineno}: {msg}")
                
                # Apply fixes
                fixer_func(filepath)
                
                # Check if fixed
                new_errors = find_syntax_errors(filepath)
                if not new_errors:
                    print(f"✅ All syntax errors fixed in {filepath}")
                else:
                    print(f"❌ Still {len(new_errors)} error(s) in {filepath}")
            else:
                print(f"\n✅ No syntax errors in {filepath}")
        else:
            print(f"\n❌ File not found: {filepath}")
    
    print("\n=== Final Compilation Test ===")
    for filepath, _ in files_to_fix:
        if os.path.exists(filepath):
            print(f"\nTesting {filepath}...")
            result = os.system(f'python -m py_compile "{filepath}" 2>&1')
            if result == 0:
                print(f"✅ {filepath} compiles successfully!")
            else:
                print(f"❌ {filepath} still has compilation errors")

if __name__ == "__main__":
    main()