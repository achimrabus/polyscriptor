import subprocess
import sys

try:
    # Get git status
    result = subprocess.run(['git', 'status', '--porcelain'], 
                          capture_output=True, 
                          text=True, 
                          cwd=r'C:\Users\Achim\Documents\TrOCR\dhlab-slavistik')
    
    print("=== Git Status (Porcelain) ===")
    print(result.stdout)
    
    # Get more detailed status
    result2 = subprocess.run(['git', 'status'], 
                           capture_output=True, 
                           text=True, 
                           cwd=r'C:\Users\Achim\Documents\TrOCR\dhlab-slavistik')
    
    print("\n=== Git Status (Detailed) ===")
    print(result2.stdout)
    
    # Check for modified files
    result3 = subprocess.run(['git', 'diff', '--name-only'], 
                           capture_output=True, 
                           text=True, 
                           cwd=r'C:\Users\Achim\Documents\TrOCR\dhlab-slavistik')
    
    print("\n=== Modified Files ===")
    print(result3.stdout if result3.stdout else "No modified tracked files")
    
    # Check for untracked files
    result4 = subprocess.run(['git', 'ls-files', '--others', '--exclude-standard'], 
                           capture_output=True, 
                           text=True, 
                           cwd=r'C:\Users\Achim\Documents\TrOCR\dhlab-slavistik')
    
    print("\n=== Untracked Files ===")
    print(result4.stdout if result4.stdout else "No untracked files")
    
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)
