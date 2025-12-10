#!/usr/bin/env python3
"""Comprehensive fixer for all trading system syntax and logic errors"""

import re
import os
import ast
import traceback
from typing import List, Tuple, Dict

class ComprehensiveSystemFixer:
    def __init__(self):
        self.errors_fixed = 0
        self.files_processed = 0
        
    def fix_dict_list_patterns(self, content: str) -> str:
        """Fix dictionary and list initialization patterns"""
        # Fix pattern: 'key': [] followed by items
        lines = content.split('\n')
        fixed_lines = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Check for dict/list initialization patterns
            if "'legs':" in line and '[]' in line:
                # Look ahead to see if next lines contain list items
                if i + 1 < len(lines) and ('{' in lines[i+1] or "{'action'" in lines[i+1]):
                    line = line.replace('[]', '[')
                    self.errors_fixed += 1
            
            # Fix empty dict followed by for loop
            if line.strip().endswith('= {') or line.strip().endswith('= ['):
                if i + 1 < len(lines):
                    next_line = lines[i+1].strip()
                    if next_line.startswith('for ') or next_line.startswith('if '):
                        line = line.rstrip()[:-1] + ('{}' if '{' in line else '[]')
                        self.errors_fixed += 1
            
            # Fix values=() followed by tuple items
            if 'values=()' in line and i + 1 < len(lines):
                next_line = lines[i+1].strip()
                if not next_line.startswith(')') and next_line and not any(kw in next_line for kw in ['def', 'class', 'if', 'for', 'while']):
                    line = line.replace('values=()', 'values=(')
                    self.errors_fixed += 1
            
            fixed_lines.append(line)
            i += 1
            
        return '\n'.join(fixed_lines)
    
    def fix_unclosed_calls(self, content: str) -> str:
        """Fix unclosed function calls"""
        # Common patterns of unclosed calls
        patterns = [
            (r'(\w+)\s*=\s*(\w+\.)?(\w+)\($', r'\1 = \2\3()'),
            (r'cp\.random\.random\(\((\d+),\s*(\d+)\)$', r'cp.random.random((\1, \2))'),
            (r'time\.time\($', r'time.time()'),
            (r'\.synchronize\($', r'.synchronize()'),
            (r'\.selection\($', r'.selection()'),
            (r'\.get_children\($', r'.get_children()'),
            (r'\.calculate_portfolio_value\($', r'.calculate_portfolio_value()'),
            (r'\._calculate_returns\($', r'._calculate_returns()'),
        ]
        
        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
            
        return content
    
    def fix_datetime_patterns(self, content: str) -> str:
        """Fix datetime calculation patterns"""
        # Fix pattern: (obj.expiry - datetime.now().days)
        content = re.sub(
            r'\((\w+\.expiry)\s*-\s*datetime\.now\(\)\.days\)',
            r'(\1 - datetime.now()).days',
            content
        )
        
        # Fix pattern: (datetime.now() - timedelta(days=x)).strftime
        content = re.sub(
            r'\(datetime\.now\(\)\s*-\s*timedelta\(days=([^)]+)\)\)\.strftime',
            r'(datetime.now() - timedelta(days=\1)).strftime',
            content
        )
        
        return content
    
    def fix_mathematical_expressions(self, content: str) -> str:
        """Fix mathematical expressions with mismatched parentheses"""
        # Fix pattern: (1 + x ** y - z) should be (1 + x) ** y - z
        content = re.sub(
            r'\(1\s*\+\s*(\w+)\.mean\(\)\s*\*\*\s*(\d+)\s*-\s*1\)',
            r'(1 + \1.mean()) ** \2 - 1',
            content
        )
        
        # Fix min/max with missing parentheses
        content = re.sub(
            r'max\(([^,]+),\s*min\(([^,]+),\s*([^)]+)\)',
            r'max(\1, min(\2, \3))',
            content
        )
        
        return content
    
    def fix_list_comprehensions(self, content: str) -> str:
        """Fix list comprehension syntax errors"""
        # Fix pattern: [c for c in list] on multiple lines
        lines = content.split('\n')
        fixed_lines = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Check for incomplete list comprehension
            if '[c for c in' in line and line.strip().endswith(']'):
                # Look at next lines for conditions
                if i + 1 < len(lines) and 'if ' in lines[i+1]:
                    # Merge the lines
                    line = line.rstrip()[:-1]  # Remove ]
                    j = i + 1
                    while j < len(lines) and (lines[j].strip().startswith('if ') or 
                                            lines[j].strip().startswith('and ') or
                                            lines[j].strip().endswith('and')):
                        line += ' ' + lines[j].strip()
                        j += 1
                    line += ']'
                    fixed_lines.append(line)
                    i = j
                    continue
            
            fixed_lines.append(line)
            i += 1
            
        return '\n'.join(fixed_lines)
    
    def fix_fstring_errors(self, content: str) -> str:
        """Fix f-string formatting errors"""
        # Fix pattern: f"{abs(min(0, x):.3f}" missing closing paren
        content = re.sub(
            r'f"\{abs\(min\(([^,]+),\s*([^)]+)\):([\d.]+)f\}"',
            r'f"{abs(min(\1, \2)):\3f}"',
            content
        )
        
        return content
    
    def fix_specific_file_issues(self, filepath: str, content: str) -> str:
        """Fix file-specific issues"""
        filename = os.path.basename(filepath)
        
        if filename == 'enhanced_ultimate_engine.py':
            # Fix the specific issue at line 1884
            content = content.replace(
                "'optimal_exit_conditions': []\n                            '50% profit target',",
                "'optimal_exit_conditions': [\n                            '50% profit target',"
            )
            
        elif filename == 'enhanced_trading_gui.py':
            # Fix control flow issues
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if i > 0 and 'else:' in line and lines[i-1].strip() == '':
                    # Check if there's a proper if/elif before
                    found_if = False
                    for j in range(max(0, i-10), i):
                        if 'if ' in lines[j] or 'elif ' in lines[j]:
                            found_if = True
                            break
                    if not found_if:
                        lines[i] = '        # ' + line.strip()  # Comment out invalid else
            content = '\n'.join(lines)
            
        elif filename == 'FINAL_ULTIMATE_COMPLETE_SYSTEM.py':
            # Fix the extra parenthesis issue
            content = content.replace(
                'returns = portfolio_manager._calculate_returns())',
                'returns = portfolio_manager._calculate_returns()'
            )
            
        return content
    
    def validate_python_syntax(self, content: str) -> List[str]:
        """Validate Python syntax and return list of errors"""
        errors = []
        try:
            ast.parse(content)
        except SyntaxError as e:
            errors.append(f"Line {e.lineno}: {e.msg}")
            if e.text:
                errors.append(f"  Problem: {e.text.strip()}")
        except Exception as e:
            errors.append(f"Parse error: {str(e)}")
        return errors
    
    def fix_file(self, filepath: str) -> Tuple[bool, List[str]]:
        """Fix a single file and return success status and remaining errors"""
        print(f"\nProcessing {os.path.basename(filepath)}...")
        self.files_processed += 1
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Apply all fixes
            content = self.fix_dict_list_patterns(content)
            content = self.fix_unclosed_calls(content)
            content = self.fix_datetime_patterns(content)
            content = self.fix_mathematical_expressions(content)
            content = self.fix_list_comprehensions(content)
            content = self.fix_fstring_errors(content)
            content = self.fix_specific_file_issues(filepath, content)
            
            # Validate syntax
            errors = self.validate_python_syntax(content)
            
            if content != original_content:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"  ✅ Fixed {self.errors_fixed} issues")
            else:
                print(f"  ℹ️  No changes needed")
            
            if errors:
                print(f"  ⚠️  {len(errors)} syntax errors remain")
                return False, errors
            else:
                print(f"  ✅ No syntax errors!")
                return True, []
                
        except Exception as e:
            print(f"  ❌ Error processing file: {e}")
            return False, [str(e)]
    
    def fix_all_systems(self, systems: List[str]) -> Dict[str, Tuple[bool, List[str]]]:
        """Fix all trading systems"""
        results = {}
        
        print("=== Comprehensive System Fixer ===")
        print(f"Processing {len(systems)} systems...\n")
        
        for system in systems:
            if os.path.exists(system):
                success, errors = self.fix_file(system)
                results[system] = (success, errors)
            else:
                print(f"⚠️  {system} not found")
                results[system] = (False, ["File not found"])
        
        print(f"\n=== Summary ===")
        print(f"Files processed: {self.files_processed}")
        print(f"Total fixes applied: {self.errors_fixed}")
        
        successful = sum(1 for success, _ in results.values() if success)
        print(f"Successfully fixed: {successful}/{len(results)}")
        
        if successful < len(results):
            print("\nRemaining issues:")
            for system, (success, errors) in results.items():
                if not success and errors:
                    print(f"\n{os.path.basename(system)}:")
                    for error in errors[:3]:  # Show first 3 errors
                        print(f"  - {error}")
        
        return results


def main():
    """Main function"""
    systems = [
        "src/misc/enhanced_ultimate_engine.py",
        "src/misc/enhanced_trading_gui.py",
        "src/misc/FINAL_ULTIMATE_COMPLETE_SYSTEM.py",
        "src/misc/ULTIMATE_AI_TRADING_SYSTEM_COMPLETE.py",
        "src/misc/ULTIMATE_COMPLEX_TRADING_GUI.py"
    ]
    
    fixer = ComprehensiveSystemFixer()
    results = fixer.fix_all_systems(systems)
    
    # Return results for testing
    return results


if __name__ == "__main__":
    main()