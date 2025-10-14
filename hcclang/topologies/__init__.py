# SPDX-License-Identifier: GPL-2.0-only

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .topo_tools import Topology, reverse_topology, binarize_topology
from . import generic
from . import cm384
from . import nvidia
from .future_topologies.distributed import distributed_fully_connected

# 导出常用的拓扑创建函数
from .generic import ring, line, star, fully_connected, hub_and_spoke
from .cm384 import cm384, cm384_full