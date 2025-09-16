#!/usr/bin/env python3
"""
VERL Installation Validation Script

This script validates that VERL is correctly installed and functional.
Run this after installing VERL to ensure everything is working properly.
"""

import sys
import importlib
from typing import Tuple


def check_import(module_name: str, description: str = None) -> Tuple[bool, str]:
    """
    Check if a module can be imported successfully.

    Args:
        module_name: Name of the module to import
        description: Optional description for the module

    Returns:
        Tuple of (success, message)
    """
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "unknown")
        desc = description or module_name
        return True, f"✓ {desc} imported successfully (version: {version})"
    except ImportError as e:
        desc = description or module_name
        return False, f"✗ {desc} import failed: {e}"
    except Exception as e:
        desc = description or module_name
        return False, f"✗ {desc} error: {e}"


def check_verl_installation() -> bool:
    """
    Comprehensive VERL installation validation.

    Returns:
        True if validation passes, False otherwise
    """
    print("=== VERL Installation Validation ===\n")

    # Core VERL import
    success, message = check_import("verl", "VERL core")
    print(message)
    if not success:
        return False

    # Get VERL version
    try:
        import verl

        print(f"✓ VERL version: {verl.__version__}")
    except Exception as e:
        print(f"⚠ Could not get VERL version: {e}")

    # Check key VERL modules
    modules_to_check = [
        ("verl.trainer", "VERLTrainer"),
        ("verl.models", "VERLModel"),
        ("verl.data", "VERL Data utilities"),
        ("verl.utils", "VERL utilities"),
    ]

    print("\n=== Checking VERL Modules ===")
    all_modules_ok = True

    for module_name, description in modules_to_check:
        success, message = check_import(module_name, description)
        print(message)
        if not success:
            all_modules_ok = False

    # Check dependencies
    print("\n=== Checking Dependencies ===")
    dependencies = [
        ("torch", "PyTorch"),
        ("transformers", "Hugging Face Transformers"),
        ("datasets", "Hugging Face Datasets"),
        ("accelerate", "Accelerate"),
        ("peft", "PEFT"),
        ("trl", "TRL"),
    ]

    for dep_name, description in dependencies:
        success, message = check_import(dep_name, description)
        print(message)
        if not success:
            all_modules_ok = False

    # GPU availability check
    print("\n=== Checking GPU Availability ===")
    try:
        import torch

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_device)
            print(f"✓ CUDA available: {gpu_count} GPU(s)")
            print(f"✓ Current GPU: {gpu_name}")
        else:
            print("⚠ CUDA not available - training will use CPU")
    except Exception as e:
        print(f"⚠ GPU check failed: {e}")

    # Test basic VERL functionality
    print("\n=== Testing Basic VERL Functionality ===")
    try:
        # Try to create a basic VERL configuration
        from verl.config import VERLConfig  # noqa: F401

        print("✓ VERLConfig can be imported")
    except ImportError:
        print("⚠ VERLConfig import failed - may not be available in this version")
    except Exception as e:
        print(f"⚠ VERLConfig test failed: {e}")

    # Summary
    print("\n=== Validation Summary ===")
    if all_modules_ok:
        print("✓ VERL installation validation PASSED!")
        print("✓ All core modules are available")
        print("✓ Ready for VERL training!")
        return True
    else:
        print("✗ VERL installation validation FAILED!")
        print("✗ Some modules are missing or have issues")
        print("✗ Please check the installation and try again")
        return False


def main():
    """Main validation function."""
    try:
        success = check_verl_installation()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nValidation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error during validation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
