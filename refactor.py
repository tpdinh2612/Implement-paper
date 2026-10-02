import os
import re

root_dir = r"e:\re implement"

replacements = [
    ('PeekingFutureConfig', 'PeekingFutureConfig'),
    ('PeekingFutureInference', 'PeekingFutureInference'),
    ('ToyPeekingFuture', 'ToyPeekingFuture'),
    ('methods.peeking_future', 'methods.peeking_future'),
    ('methods/peeking_future', 'methods/peeking_future'),
    ('run_peeking_future', 'run_peeking_future'),
    ('peeking_future_config', 'peeking_future_config'),
    ('peeking_future_baseline', 'peeking_future_baseline'),
    ('peeking_future_architecture', 'peeking_future_architecture'),
    ('"peeking-future"', '"peeking-future"'),
    ("'peeking-future'", "'peeking-future'"),
]

def replace_content(content):
    for old, new in replacements:
        content = content.replace(old, new)
        
    # Replace exact matches of "peeking-future" as a word (for variables, etc.)
    content = re.sub(r'\bparco\b', 'peeking_future', content)
    
    # Replace remaining "PEEKING-FUTURE" (usually in docstrings, comments, print statements)
    content = re.sub(r'\bPARCO\b', 'PEEKING-FUTURE', content)
    
    return content

# 1. Rename files and directories first
for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
    if '.git' in dirpath or '__pycache__' in dirpath or '.venv' in dirpath:
        continue
    
    for filename in filenames:
        if 'peeking-future' in filename.lower():
            old_path = os.path.join(dirpath, filename)
            if filename.endswith('.py'):
                new_name = filename.replace('peeking-future', 'peeking_future').replace('PEEKING-FUTURE', 'PeekingFuture')
            else:
                new_name = filename.replace('peeking-future', 'peeking-future').replace('PEEKING-FUTURE', 'PEEKING-FUTURE')
            os.rename(old_path, os.path.join(dirpath, new_name))
            
    for i in range(len(dirnames)):
        dirname = dirnames[i]
        if 'peeking-future' in dirname.lower():
            old_path = os.path.join(dirpath, dirname)
            new_name = dirname.replace('peeking-future', 'peeking_future').replace('PEEKING-FUTURE', 'PeekingFuture')
            os.rename(old_path, os.path.join(dirpath, new_name))
            dirnames[i] = new_name

# 2. Replace content inside files
for dirpath, dirnames, filenames in os.walk(root_dir):
    if '.git' in dirpath or '__pycache__' in dirpath or '.venv' in dirpath:
        continue
    for filename in filenames:
        if filename.endswith(('.py', '.md', '.json', '.txt', '.csv')):
            filepath = os.path.join(dirpath, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                continue
                
            new_content = replace_content(content)
            
            if content != new_content:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                print(f"Updated content in {filepath}")

print("Done renaming and replacing!")
