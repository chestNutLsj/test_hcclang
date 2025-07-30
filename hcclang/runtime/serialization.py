# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from hcclang.core.algorithm import Algorithm, Step
from hcclang.topologies import Topology
from hcclang.solver.instance import Instance
from hcclang.core.collectives import Collective, Chunk

import json
import warnings

def _hccl_object_hook(o):
    if not 'hccl_type' in o:
        return o
    if o['hccl_type'] == 'algorithm':
        input_map = { int(k): set(v) for k, v in o['input_map'].items() }
        output_map = { int(k): set(v) for k, v in o['output_map'].items() }
        return Algorithm(o['name'], o['collective'], o['topology'], o['instance'], o['steps'], input_map, output_map)
    if o['hccl_type'] == 'step':
        sends = [(addr, src, dst) for addr, src, dst in o['sends']]
        return Step(o['rounds'], sends)
    if o['hccl_type'] == 'collective':
        triggers = { (int(r), int(c)): v for r, rmap in o['triggers'].items() for c, v in rmap.items() }
        return Collective(o['name'], o['nodes'], o['chunks'], triggers, o['runtime_name'])
    if o['hccl_type'] == 'chunk':
        pre = set(o['pre'])
        post = set(o['post'])
        return Chunk(pre, post, o['addr'])
    if o['hccl_type'] == 'topology':
        return Topology(o['name'], o['links'], o['switches'])
    if o['hccl_type'] == 'instance':
        return Instance(o['steps'], o['extra_rounds'], o['chunks'], o['pipeline'], o['extra_memory'], o['allow_exchange'])
    warnings.warn('Unhandled hccl_type in JSON')

def HCCLDecoder():
    return json.JSONDecoder(object_hook=_hccl_object_hook)

class HCCLEncoder(json.JSONEncoder):
    def __init__(self):
        super().__init__()
    
    def default(self, o):
        if isinstance(o, Algorithm):
            input_map = { k: list(v) for k, v in o.input_map.items() }
            output_map = { k: list(v) for k, v in o.output_map.items() }
            return {
                'hccl_type': 'algorithm',
                'name': o.name,
                'instance': o.instance,
                'input_map': input_map,
                'output_map': output_map,
                'steps': o.steps,
                'collective': o.collective,
                'topology': o.topology,
            }
        if isinstance(o, Step):
            return {
                'hccl_type': 'step',
                'rounds': o.rounds,
                'sends': o.sends,
            }
        if isinstance(o, Collective):
            triggers = {}
            for (r, c), v in o._triggers.items():
                if not r in triggers:
                    triggers[r] = {}
                triggers[r][c] = v
            return {
                'hccl_type': 'collective',
                'name': o.name,
                'nodes': o.num_nodes,
                'chunks': o._chunks,
                'triggers': triggers,
                'runtime_name': o.runtime_name,
            }
        if isinstance(o, Chunk):
            return {
                'hccl_type': 'chunk',
                'pre': list(o.precondition),
                'post': list(o.postcondition),
                'addr': o.address,
            }
        if isinstance(o, Topology):
            return {
                'hccl_type': 'topology',
                'name': o.name,
                'switches': o.switches,
                'links': o.links,
            }
        if isinstance(o, Instance):
            return {
                'hccl_type': 'instance',
                'steps': o.steps,
                'extra_rounds': o.extra_rounds,
                'chunks': o.chunks,
                'pipeline': o.pipeline,
                'extra_memory': o.extra_memory,
                'allow_exchange': o.allow_exchange,
            }
        return json.JSONEncoder.default(self, o)

def save_hccl_object(obj, filename):
    with open(filename, 'w') as f:
        f.write(HCCLEncoder().encode(obj))

def load_hccl_object(filename):
    with open(filename) as f:
        return HCCLDecoder().decode(f.read())

# Backward compatibility aliases
save_msccl_object = save_hccl_object
load_msccl_object = load_hccl_object
MSCCLEncoder = HCCLEncoder
MSCCLDecoder = HCCLDecoder
