"""
Citations (open-source):
- NetworkX: https://networkx.org/
"""
from __future__ import annotations  

import sys
import os
import pickle
from pathlib import Path

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import networkx as nx

try:
    from pnu_flow.config import SIM_CFG, PATHS
except ModuleNotFoundError:
    from config import SIM_CFG, PATHS

def build_ccis_graph() -> nx.DiGraph:
    """
    Build CCIS indoor building graph.
    """
    g = nx.DiGraph()

    # Add nodes with coordinates and capacity
    for zone in SIM_CFG.building_zones:
        g.add_node(zone,
                   capacity=SIM_CFG.zone_capacity[zone],
                   pos=SIM_CFG.zone_coords[zone])

    # Physical connections with base distances (metres)
    edges = [
        ("Main_Lobby",        "North_Wing_G",     8.0),
        ("Main_Lobby",        "South_Wing_G",    10.0),
        ("North_Wing_G",      "Elevators_G",     12.0),
        ("South_Wing_G",      "Elevators_G",      9.0),
        ("North_Wing_G",      "Student_Lounge",  14.0),
        ("South_Wing_G",      "Student_Lounge",  16.0),
        ("Elevators_G",       "Central_Stairs_G1",7.0),
        ("Central_Stairs_G1", "Elevators_F1",    11.0),
        ("Elevators_F1",      "North_Wing_F1",   10.0),
        ("Elevators_F1",      "South_Wing_F1",   10.0),
        ("North_Wing_F1",     "Quiet_Study_Area", 9.0),
        ("South_Wing_F1",     "Quiet_Study_Area", 10.0),
        ("Quiet_Study_Area",  "Central_Stairs_12", 6.0),
        ("Central_Stairs_12", "Elevators_F2",    11.0),
        ("Elevators_F2",      "North_Wing_F2",    8.0),
        ("Elevators_F2",      "South_Wing_F2",    8.5),
        ("North_Wing_F2",     "Grand_Auditorium", 12.0),
        ("South_Wing_F2",     "Grand_Auditorium", 14.0),
    ]
    
    for u, v, w in edges:
        g.add_edge(u, v, base_weight=w)
        g.add_edge(v, u, base_weight=w)
    return g

if __name__ == "__main__":
    print("Building Graph with new zone names...")
    graph = build_ccis_graph()
    
    zone_mapping = {node: i for i, node in enumerate(graph.nodes())}
    
    PATHS.artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    mapping_path = PATHS.artifacts_dir / "zone_mapping.pkl"
    with open(mapping_path, "wb") as f:
        pickle.dump(zone_mapping, f)
    
    print(f"Successfully updated: {mapping_path}")
    print("Zones found:", list(zone_mapping.keys()))