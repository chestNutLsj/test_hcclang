# SPDX-License-Identifier: GPL-2.0-only

#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Test runner for hcclang.language module tests.
"""

import os
import sys

# Add the parent directory to path to import hcclang
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

def main():
    """Run all language tests."""
    print("HCCLang Language Module Test Suite")
    print("=" * 50)
    
    # Create output directory for test results
    output_dir = os.path.join(os.path.dirname(__file__), "test_outputs")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Run import and basic tests
    print("\n--- Running Import and Basic Tests ---")
    try:
        from .test_import_and_basic import run_all_tests as run_basic_tests
        basic_success = run_basic_tests()
    except Exception as e:
        print(f"Failed to run basic tests: {e}")
        basic_success = False
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Suite Summary:")
    print(f"- Basic Tests: {'PASSED' if basic_success else 'FAILED'}")
    
    if basic_success:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 