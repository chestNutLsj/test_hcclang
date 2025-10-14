# SPDX-License-Identifier: GPL-2.0-only

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# HCCLang - A domain-specific language for HCCL collective communication algorithms
# Modified from MSCCLang for HCCL compatibility

# Core algorithm and collective definitions
from .core import *

# Language constructs
from .language import *

# Topology definitions
from .topologies import *

# Runtime code generation
from .runtime import *

# Optimization utilities (optional)
from .optimization import *

# Solver components (optional for advanced users)
from .solver import *

# Pre-defined algorithm programs
from .programs import *

__version__ = "1.0.0"
