#!/usr/bin/env python3
"""
Validation script for repository cleanliness
"""
import os
import glob
import subprocess
from pathlib import Path


def validate_repository_cleanliness():
    """
    Validate that the repository contains only essential files
    """
    print("Validating repository cleanliness...")

    # Check for common non-essential files
    patterns_to_avoid = [
        "**/*.log",
        "**/*.tmp",
        "**/*.temp",
        "**/*.bak",
        "**/*.backup",
        "**/*~",
        "**/.*~",
        "**/__pycache__/",
        "**/*.pyc",
        "**/node_modules/",
        "**/package-lock.json",
        "**/yarn.lock",
        "**/.DS_Store",
        "**/Thumbs.db",
        "**/desktop.ini",
    ]

    issues_found = []
    for pattern in patterns_to_avoid:
        files_found = glob.glob(pattern, recursive=True)
        if files_found:
            issues_found.extend(files_found)

    # Report findings
    if issues_found:
        print(f"ISSUES FOUND: {len(issues_found)} non-essential files/directories detected:")
        for issue in issues_found[:10]:  # Show first 10 issues
            print(f"  - {issue}")
        if len(issues_found) > 10:
            print(f"  ... and {len(issues_found) - 10} more")
        return False
    else:
        print("✓ Repository appears clean - no non-essential files found")
        return True


def validate_gitignore():
    """
    Validate that .gitignore is properly configured
    """
    print("\nValidating .gitignore configuration...")

    required_patterns = [
        "*.log",
        "*.tmp",
        "__pycache__/",
        "*.pyc",
        ".DS_Store",
        "node_modules/",
        "Thumbs.db"
    ]

    try:
        with open('.gitignore', 'r') as f:
            gitignore_content = f.read()

        missing_patterns = []
        for pattern in required_patterns:
            if pattern not in gitignore_content:
                missing_patterns.append(pattern)

        if missing_patterns:
            print(f"✗ Missing patterns in .gitignore: {missing_patterns}")
            return False
        else:
            print("✓ .gitignore contains all required patterns")
            return True
    except FileNotFoundError:
        print("✗ .gitignore file not found")
        return False


def validate_project_structure():
    """
    Validate that essential project structure is intact
    """
    print("\nValidating project structure...")

    essential_files = [
        "backend/main.py",
        "backend/requirements.txt",
        "backend/src/rag_agent/__init__.py",
        "backend/src/rag_agent/agent.py",
        "src/components/Chatbot/api/chatService.js",
        "specs/001-rag-integration/spec.md",
        "specs/001-rag-integration/plan.md",
        "specs/001-rag-integration/tasks.md",
    ]

    missing_files = []
    for file_path in essential_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)

    if missing_files:
        print(f"✗ Missing essential files: {missing_files}")
        return False
    else:
        print("✓ All essential files are present")
        return True


def main():
    """
    Main validation function
    """
    print("Repository Cleanup Validation")
    print("=" * 40)

    # Run all validation checks
    cleanliness_ok = validate_repository_cleanliness()
    gitignore_ok = validate_gitignore()
    structure_ok = validate_project_structure()

    print("\n" + "=" * 40)
    print("VALIDATION SUMMARY:")
    print(f"  Repository Cleanliness: {'✓ PASS' if cleanliness_ok else '✗ FAIL'}")
    print(f"  Gitignore Configuration: {'✓ PASS' if gitignore_ok else '✗ FAIL'}")
    print(f"  Project Structure: {'✓ PASS' if structure_ok else '✗ FAIL'}")

    overall_status = cleanliness_ok and gitignore_ok and structure_ok
    print(f"  Overall Status: {'✓ PASS' if overall_status else '✗ FAIL'}")

    return overall_status


if __name__ == "__main__":
    main()