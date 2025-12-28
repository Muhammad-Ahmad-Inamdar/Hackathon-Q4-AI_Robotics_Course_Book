#!/usr/bin/env python3
"""
Cleanup script for removing non-essential files from the repository
"""
import os
import glob
import shutil
from pathlib import Path


def scan_for_unused_files():
    """
    Scan the repository for unused files that should be cleaned up
    """
    patterns_to_remove = [
        # Log files
        "**/*.log",
        "**/*.tmp",
        "**/*.temp",
        "**/*.log.*",

        # Temporary files
        "**/*~",
        "**/.*~",
        "**/*.bak",
        "**/*.backup",

        # Duplicate markdown files (non-essential)
        "**/README_backup*.md",
        "**/README_old*.md",
        "**/*_backup*.md",
        "**/*_old*.md",

        # Manual ingestion scripts (if they exist outside proper locations)
        "**/manual_ingest*.py",
        "**/ingest_manual*.py",

        # Text files that are not essential documentation
        "**/temp_*.txt",
        "**/test_*.txt",
        "**/debug_*.txt",
        "**/output_*.txt",

        # Duplicate configuration files
        "**/config_backup*.json",
        "**/config_old*.json",

        # Python cache files outside the project structure
        "**/__pycache__/",
        "**/*.pyc",
        "**/*.pyo",
        "**/*.pyd",

        # Node.js cache files if present in wrong locations
        "**/node_modules/",
        "**/package-lock.json",
        "**/yarn.lock",

        # OS-specific files
        "**/.DS_Store",
        "**/Thumbs.db",
        "**/desktop.ini",
    ]

    files_to_remove = []
    for pattern in patterns_to_remove:
        files_to_remove.extend(glob.glob(pattern, recursive=True))

    # Filter out files that are actually needed
    essential_files = [
        "backend/requirements.txt",
        "backend/.env.example",
        "src/components/Chatbot/api/chatService.js",
    ]

    final_files = []
    for file_path in files_to_remove:
        if file_path not in essential_files:
            final_files.append(file_path)

    return final_files


def remove_files(file_list):
    """
    Remove the specified files and directories
    """
    removed_count = 0
    for file_path in file_list:
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Removed file: {file_path}")
                removed_count += 1
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
                print(f"Removed directory: {file_path}")
                removed_count += 1
        except Exception as e:
            print(f"Error removing {file_path}: {str(e)}")

    return removed_count


def update_gitignore():
    """
    Update .gitignore to exclude temporary files and logs
    """
    gitignore_content = """
# Logs
*.log
*.log.*
log/
logs/

# Temporary files
*.tmp
*.temp
*~
.*~
*.bak
*.backup

# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.venv/
pip-log.txt
pip-delete-this-directory.txt
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.log
.git/refs/replace/
.pytest_cache/

# Node.js
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*
.pnp/
.pnp.js
.next/
.nuxt/
dist/
build/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
desktop.ini

# Environment specific
.env.local
.env.development.local
.env.test.local
.env.production.local
.env.*

# Backup files
*~
*.bak
*.backup
"""

    with open('.gitignore', 'a') as f:
        f.write(gitignore_content)

    print("Updated .gitignore file")


def main():
    """
    Main cleanup function
    """
    print("Scanning for unused files...")
    files_to_remove = scan_for_unused_files()

    if not files_to_remove:
        print("No files to remove.")
        return 0

    print(f"Found {len(files_to_remove)} files/directories to remove:")
    for file_path in files_to_remove:
        print(f"  - {file_path}")

    response = input("\nDo you want to proceed with removal? (yes/no): ")
    if response.lower() in ['yes', 'y']:
        removed_count = remove_files(files_to_remove)
        update_gitignore()
        print(f"\nCleanup completed. Removed {removed_count} files/directories.")
        return removed_count
    else:
        print("Cleanup cancelled.")
        return 0


if __name__ == "__main__":
    main()