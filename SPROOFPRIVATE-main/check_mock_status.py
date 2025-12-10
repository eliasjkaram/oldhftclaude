#!/usr/bin/env python3
"""
Mock Implementation Status Checker
==================================
Quickly identify and count mock implementations across the codebase.
"""

import os
import re
from collections import defaultdict
from datetime import datetime

class MockChecker:
    def __init__(self, root_dir='src'):
        self.root_dir = root_dir
        self.mock_patterns = {
            'random_returns': r'random\.(uniform|choice|randint|random)',
            'none_returns': r'return\s+None',
            'todo_comments': r'#\s*TODO',
            'placeholder': r'placeholder|PLACEHOLDER',
            'mock_mode': r'mock_mode|test_mode',
            'hardcoded': r'return\s+["\']?(bullish|bearish|buy|sell)["\']?',
            'not_implemented': r'raise\s+NotImplementedError',
            'pass_only': r'^\s*pass\s*$',
            'dummy': r'dummy|DUMMY|fake|FAKE',
            'simulate': r'simulat(e|ed|ing)|synthetic'
        }
        self.results = defaultdict(list)
        self.file_count = defaultdict(int)
        
    def check_file(self, filepath):
        """Check a single file for mock patterns"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
                
            for pattern_name, pattern in self.mock_patterns.items():
                matches = list(re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE))
                
                for match in matches:
                    # Find line number
                    line_num = content[:match.start()].count('\n') + 1
                    line_content = lines[line_num - 1].strip()
                    
                    # Skip comments and docstrings
                    if line_content.startswith('#') and pattern_name != 'todo_comments':
                        continue
                        
                    self.results[pattern_name].append({
                        'file': filepath,
                        'line': line_num,
                        'content': line_content[:80] + '...' if len(line_content) > 80 else line_content
                    })
                    
            if matches:
                self.file_count[filepath] = len(matches)
                
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            
    def scan_directory(self):
        """Scan entire directory structure"""
        total_files = 0
        
        for root, dirs, files in os.walk(self.root_dir):
            # Skip __pycache__ directories
            dirs[:] = [d for d in dirs if d != '__pycache__']
            
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    self.check_file(filepath)
                    total_files += 1
                    
        return total_files
        
    def get_critical_files(self):
        """Identify critical files with most mocks"""
        critical_paths = ['core', 'data', 'execution', 'risk', 'ml']
        critical_files = []
        
        for filepath, count in sorted(self.file_count.items(), key=lambda x: x[1], reverse=True):
            for critical in critical_paths:
                if critical in filepath:
                    critical_files.append((filepath, count))
                    break
                    
        return critical_files[:10]  # Top 10 critical files
        
    def generate_report(self):
        """Generate comprehensive report"""
        print("=" * 80)
        print("MOCK IMPLEMENTATION STATUS REPORT")
        print("=" * 80)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Directory: {self.root_dir}")
        print()
        
        # Summary statistics
        total_mocks = sum(len(items) for items in self.results.values())
        print(f"📊 SUMMARY")
        print(f"Total mock implementations found: {total_mocks}")
        print(f"Files with mocks: {len(self.file_count)}")
        print()
        
        # Pattern breakdown
        print("📈 MOCK PATTERNS BREAKDOWN")
        for pattern, items in sorted(self.results.items(), key=lambda x: len(x[1]), reverse=True):
            if items:
                print(f"\n{pattern.replace('_', ' ').title()}: {len(items)} occurrences")
                # Show top 3 examples
                for item in items[:3]:
                    print(f"  {item['file']}:{item['line']}")
                    print(f"    {item['content']}")
                if len(items) > 3:
                    print(f"  ... and {len(items) - 3} more")
                    
        print("\n" + "=" * 80)
        print("🔥 CRITICAL FILES (Most Mocks)")
        print("=" * 80)
        
        critical_files = self.get_critical_files()
        for filepath, count in critical_files:
            print(f"{count:3d} mocks: {filepath}")
            
        print("\n" + "=" * 80)
        print("🎯 PRIORITY FIXES")
        print("=" * 80)
        
        # Identify priority fixes
        priority_fixes = []
        
        # Check for None returns in data providers
        for item in self.results['none_returns']:
            if 'data' in item['file'] and 'fetch' in item['content'].lower():
                priority_fixes.append(f"DATA: {item['file']}:{item['line']} - {item['content']}")
                
        # Check for random in ML predictions
        for item in self.results['random_returns']:
            if 'ml' in item['file'] or 'predict' in item['content'].lower():
                priority_fixes.append(f"ML: {item['file']}:{item['line']} - {item['content']}")
                
        for fix in priority_fixes[:10]:
            print(f"  {fix}")
            
        print("\n" + "=" * 80)
        print("📋 QUICK ACTIONS")
        print("=" * 80)
        print("1. Fix data fetching:")
        print("   vim src/data/market_data/enhanced_data_provider.py")
        print("2. Remove synthetic data dependency:")
        print("   grep -n 'use_synthetic' src/data/market_data/enhanced_data_provider.py")
        print("3. Train ML models:")
        print("   python src/ml/train_models.py")
        print("4. Check specific pattern:")
        print(f"   grep -r 'return None' {self.root_dir}/ | grep -v __pycache__")
        
    def export_json(self, filename='mock_analysis.json'):
        """Export results to JSON for further analysis"""
        import json
        
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_mocks': sum(len(items) for items in self.results.values()),
                'files_affected': len(self.file_count),
                'patterns': {k: len(v) for k, v in self.results.items()}
            },
            'critical_files': self.get_critical_files(),
            'details': self.results
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
            
        print(f"\nDetailed analysis exported to: {filename}")

def main():
    """Run mock checker"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Check for mock implementations')
    parser.add_argument('--dir', default='src', help='Directory to scan')
    parser.add_argument('--export', action='store_true', help='Export to JSON')
    parser.add_argument('--pattern', help='Check specific pattern only')
    
    args = parser.parse_args()
    
    checker = MockChecker(args.dir)
    
    if args.pattern:
        # Check only specific pattern
        checker.mock_patterns = {args.pattern: checker.mock_patterns.get(args.pattern, args.pattern)}
    
    print(f"🔍 Scanning {args.dir} directory...")
    total_files = checker.scan_directory()
    print(f"✅ Scanned {total_files} Python files\n")
    
    checker.generate_report()
    
    if args.export:
        checker.export_json()

if __name__ == "__main__":
    main()