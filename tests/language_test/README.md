# HCCLang Language Module Tests

This directory contains comprehensive tests for the `hcclang.language` module, which is the core language component for defining collective communication operations and their semantics.

## Test Structure

### Test Files

- **`test_corrected_imports.py`** - Basic functionality tests including module imports, program creation, and core operations
- **`test_advanced_functionality.py`** - Advanced tests covering collective operations, XML generation, and IR lowering
- **`run_all_tests.py`** - Main test runner that executes all test suites and generates reports

### Legacy Test Files

- `test_import_and_basic.py` - Initial import testing (superseded by corrected imports)
- `test_basic_functionality.py` - Initial basic tests (superseded by corrected imports)
- `test_summary.py` - Test summary generator (superseded by run_all_tests.py)
- `run_tests.py` - Simple test runner (superseded by run_all_tests.py)

## Running Tests

### Quick Test Run

To run all tests and generate a comprehensive report:

```bash
cd tests/language_test
python run_all_tests.py
```

### Individual Test Runs

Run specific test suites:

```bash
# Basic functionality tests
python test_corrected_imports.py

# Advanced functionality tests  
python test_advanced_functionality.py
```

## Test Results

Test results and reports are saved in the `test_outputs/` directory:

- `final_test_report.md` - Comprehensive test report with all results
- Individual test outputs are also captured

## Module Changes Validated

The tests validate the following changes made to migrate from MSCCL to HCCL:

### 1. Class and Function Renaming
- ✅ `MSCCLProgram` → `HCCLProgram`
- ✅ Fixed `__enter__` method to return `self`

### 2. Comment and String Updates
- ✅ "MSCCL-IR" → "HCCL-IR" in comments
- ✅ "MSCCLang" → "HCCLang" in error messages

### 3. Functionality Validation
- ✅ Program creation and context management
- ✅ Buffer operations and management
- ✅ Collective operations (AllReduce, AllGather, ReduceScatter, AllToAll)
- ✅ Chunk operations (split, group, copy, reduce)
- ✅ InstructionDAG functionality
- ✅ XML generation and IR lowering
- ✅ Integration with topologies module

## Dependencies

The tests require:
- Python 3.12+ (using dev312 conda environment)
- hcclang package modules:
  - `hcclang.language` (core module being tested)
  - `hcclang.topologies` (for topology integration)
- Standard library modules: `os`, `sys`, `datetime`, `subprocess`

## Test Coverage

### Basic Tests (test_corrected_imports.py)
1. Module imports and dependencies
2. Buffer enum functionality from IR module
3. Available collective types creation
4. Program context manager usage
5. Chunk operations within program context
6. InstructionDAG creation and basic functionality

### Advanced Tests (test_advanced_functionality.py)
1. Simple AllReduce program with actual operations
2. Copy operations between different ranks
3. Reduce operations between chunks
4. XML generation from programs
5. Program lowering to IR
6. Advanced buffer management with AllGather

## Integration Notes

These tests ensure that the language module correctly integrates with:
- **Topologies Module**: Tests use `hcclang.topologies.generic` for creating network topologies
- **IR Module**: Tests validate IR generation and XML output functionality
- **Buffer System**: Tests validate the buffer management and chunk operations

All tests pass successfully, confirming that the module is ready for production use with the HCCL naming conventions. 