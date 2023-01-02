# -*- coding: utf-8 -*-
"""
Created on Thu May 19 15:58:01 2022

@author: Chris

Main Inputs
1. OSMnx graph of region, cleaned for vehicle roads only and with metrics of 
    travel time, distance, cost and ferry route costs added.
2. Dataframe of resources with the nearest OSMnx node index added.

For the nodes given by the dataframe, calculates the distance matrices for 
    all nodes and metrics given.
"""

import pandas as pd
import numpy as np
import networkx as nx
import time 
from pandas.api.types import is_numeric_dtype
from tqdm import tqdm


#%%

def mean_str(col):
    if is_numeric_dtype(col):
        return col.mean()
    else:
          if col.nunique() == 1:
              return [i for i in col.unique() if i==i ][0]
          return np.nan

def get_node_to_int(nodes):
    return dict(zip(sorted(nodes),[i for i in range(len(nodes))]))

def distt(G,out,node,nodes,weight,node_to_int):
    to_add = dict(nx.shortest_path_length(G,source=node,weight=weight))
    to_add = [to_add[i] for i in nodes]
    if weight == "cost":
        to_add = [round(i*100) for i in to_add]
    out[node_to_int[node],:] = to_add

def get_weighted_dists(G, isle, nodes, weight):
    start = time.time()
    node_to_int = get_node_to_int(nodes)
    out = np.zeros([len(nodes)]*2)
    for node in tqdm(nodes):
        distt(G,out,node,nodes,weight,node_to_int)
    print("\n",isle," FINISHED IN ",time.time()-start,"\n\n")

    return out

#%%

coll_area = "Islay and Jura"

df = pd.read_csv("biomass_resource_df.csv",
                 index_col = 0)
df = df.loc[df.coll_area==coll_area]

GRAPH = "islay_jura_osmnx_graphs.pickle"

graphs = pd.read_pickle(GRAPH)

short_coll_areas = df.groupby("nearest_node").first().coll_area.value_counts().sort_values().index

m = {}

G = graphs["graph_osgb"]
# NODE DICT IS JUST IN ORDER OF NODES SMALLEST TO LARGEST
nodes = sorted(df.nearest_node.unique())

for i in ["time_s","length_ferry","cost_ferry","cost"]:
    m[i]= {}
    m[i][coll_area] = get_weighted_dists(G,coll_area,nodes,i)

pd.to_pickle(m,"dist_matrices_by_metric.pickle")
 
