"""LOOK observation/writeback sites; no changes to the trained MHD topology."""
from __future__ import annotations

import torch

PROTOCOL = 'joint_sequential_optional_v1'


def correction_sites(graph):
    position = getattr(graph, 'fusion_position', None)
    stages = ['input', 'stem', 'layer1', 'layer2', 'layer3', 'layer4', 'feature']
    if position not in stages:
        return list(graph.correction_nodes)
    return [f'joint_{s}' for s in stages[:stages.index(position) + 1]] + list(graph.correction_nodes)


def members(name):
    if name.startswith('joint_'):
        stage = name.removeprefix('joint_')
        return (f'oct_{stage}', f'cfp_{stage}')
    return (name,)


def site_level(graph, name):
    if name == 'joint_input':
        return -1  # Before flatten-eyes and the first convolution.
    levels = [graph.node_level_map[n] for n in members(name)]
    if len(set(levels)) != 1:
        raise ValueError(f'Joint members must be at one forward level: {name}')
    return levels[0]


def member_shapes(graph, name):
    return tuple(tuple(graph.get_node_by_name(n).feature_message.current_state.shape[1:])
                 for n in members(name))


def read_site(graph, name):
    states = [graph.get_node_by_name(n).feature_message.current_state for n in members(name)]
    if len(states) == 1:
        return states[0]
    if name == 'joint_input':
        if any(x.ndim != 5 or x.shape[1] != 2 for x in states):
            raise ValueError('Joint input expects paired [B,2,C,H,W] tensors')
        states = [x.reshape(-1, *x.shape[2:]) for x in states]
    if any(x.shape[0] != states[0].shape[0] or x.shape[2:] != states[0].shape[2:] for x in states):
        raise ValueError(f'Incompatible paired states at {name}')
    return torch.cat(states, dim=1)


def write_site(graph, name, value):
    names = members(name)
    if len(names) == 1:
        graph.get_node_by_name(name).feature_message.current_state = value
        return
    old = [graph.get_node_by_name(n).feature_message.current_state for n in names]
    channels = [x.shape[2] if name == 'joint_input' else x.shape[1] for x in old]
    parts = value.split(channels, dim=1)
    for n, part, previous in zip(names, parts, old):
        graph.get_node_by_name(n).feature_message.current_state = part.reshape(previous.shape)


def resolve_sites(graph, configured):
    available = correction_sites(graph)
    selected = available if configured == ['all_available'] else list(configured)
    if len(set(selected)) != len(selected) or any(n not in available for n in selected):
        raise ValueError(f'Invalid LOOK sites: {selected}')
    if selected != [n for n in available if n in selected]:
        raise ValueError('LOOK sites must follow forward order')
    return selected
