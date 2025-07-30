#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import hcclang.core as core
import hcclang.topologies as topologies
import hcclang.runtime as runtime
import hcclang.optimization as optimization
import hcclang.solver as solver
import hcclang.programs as programs
from hcclang.cli import *

import argparse
import argcomplete
import sys

def main():
    parser = argparse.ArgumentParser('hcclang')

    cmd_parsers = parser.add_subparsers(title='command', dest='command')
    cmd_parsers.required = True

    handlers = []
    handlers.append(make_solvers(cmd_parsers))
    handlers.append(make_composers(cmd_parsers))
    handlers.append(make_distributors(cmd_parsers))
    handlers.append(make_analyses(cmd_parsers))
    handlers.append(make_handle_ncclize(cmd_parsers))
    handlers.append(make_plans(cmd_parsers))

    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    
    for handler in handlers:
        if handler(args, args.command):
            break

if __name__ == '__main__':
    main()
