# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 11:47:21 2022

@author: Chris

Inputs
1. Dataframe of resources with matched OSMnx nodes, resource type and location 
    (grouped by local authority, collection_area and island).
2. Distance matrices for cost, distance, time and ferry cost.
3. Depot locations matched to nearest OSMnx nodes.

Calculates the optimised collection routes using the OR-Tools CVRP solver for 
smaller problems (<2000 nodes) and recursive DBSCAN cluster for larger (>2000
nodes) problems. Number of vehicles and total collection costs (ferries and 
distance travelled) are returned for the techno-economic model.
"""

import pandas as pd
from ortools.constraint_solver import routing_enums_pb2, pywrapcp
from ortools.linear_solver import pywraplp
import networkx as nx
import numpy as np
from time import time as time
from sklearn.cluster import DBSCAN
from copy import copy,deepcopy
from ast import literal_eval
    

#%% FROM ORTOOLS WALKTHROUGH - https://developers.google.com/optimization/routing/cvrp

def list_solutions(data, manager, routing, solution):
    routes = {}
    route_dists = {}
    route_durations = {}
    route_ferries = {}

    for vehicle_id in range(data['num_vehicles']):
        routes[vehicle_id] = pd.Series(dtype=float)
        index = routing.Start(vehicle_id)
        route_load = 0
        route_duration = 0
        route_dist = 0
        route_ferry = 0
        
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_load += data['demands'][node_index]
            previous_index = index
            previous_node_index = node_index
            
            index = solution.Value(routing.NextVar(index))
            node_index = manager.IndexToNode(index)
            ind = manager.IndexToNode(index)
            routes[vehicle_id].loc[ind] = route_load
            route_duration += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)
            route_dist += data["new_dists"][previous_node_index,node_index]
            route_ferry += data["new_ferry_costs"][previous_node_index,node_index]
            
        route_durations[vehicle_id] = route_duration
        route_dists[vehicle_id] = route_dist
        route_ferries[vehicle_id] = route_ferry
        routes[vehicle_id].sort_values(inplace=True)
    return routes, route_durations, route_dists, route_ferries

def cvrp_solver(data, area, limit=False,time_constraint = 8*3600):
    """Solve the CVRP problem."""
    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['depot'])
    
    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)
    
    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]
    
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    
    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    
    # Add Capacity constraint.
    def demand_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]
    
    demand_callback_index = routing.RegisterUnaryTransitCallback(
        demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data['vehicle_capacities'],  # vehicle maximum capacities
        True,  # start cumul to zero
        'Capacity')
    
    # Add time constraint.
    dimension_name = 'Time'
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        time_constraint,  # vehicle maximum travel distance
        True,  # start cumul to zero
        dimension_name)
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    distance_dimension.SetGlobalSpanCostCoefficient(100)
    
    # Allow to drop nodes.
    penalty = 10**10
    for node in range(1, len(data['distance_matrix'])):
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty)
    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    
    search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC)
    if limit:
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        search_parameters.time_limit.FromSeconds(limit)
        
    
    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)
    
    # Return solution
    if solution:
        print("MAIN CVRP solved for {0}".format(area))
        return list_solutions(data, manager, routing, solution)
    else:
        print("NO MAIN CVRP solution for {0}".format(area))
        return [np.nan] * 4
    
def get_route_length(route,array,node_to_int_array=False):
    """can handle node or int list but if node need to include int map"""
    length = []
    # ferry_count = 0
    if node_to_int_array:
        assert len(node_to_int_array)==array.shape[0]
        for i in range(len(route)-1):
            from_node = node_to_int_array[route[i]]
            to_node = node_to_int_array[route[i+1]]
            length.append(array[from_node, to_node])
        return length
    for i in range(len(route)-1):
        node = route[i]
        next_node = route[i+1]
        length.append(array[node, next_node])
    return sum(length)
    

def salesperson_solver_no_return(data,label):
    """data needs dist matrix, vehicles, and the depot (start/finish)-
    to cancel the return trip to the depot, the first node distances aer
    set to zero"""
    
    # https://or.stackexchange.com/questions/6174/travelling-salesman-problem-variant-without-returning-to-the-starting-point
    data["distance_matrix"][0,:] = 0
    data["distance_matrix"][:,0] = 0
    data["depot"] = 0
    data["num_vehicles"] = 1
    
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                       data['num_vehicles'], data['depot'])
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]
    
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    #adding penalty for dropped 
    

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)
    def get_routes(solution, routing, manager):
        """Get vehicle routes from a solution and store them in an array."""
        # Get vehicle routes and store them in a two dimensional array whose
        # i,j entry is the jth location visited by vehicle i along its route.
        for route_nbr in range(routing.vehicles()):
          index = routing.Start(route_nbr)
          route = [manager.IndexToNode(index)]
          while not routing.IsEnd(index):
            index = solution.Value(routing.NextVar(index))
            route.append(manager.IndexToNode(index))
          # routes.append(route)
        return route
    if solution:
        route = {label:get_routes(solution, routing, manager)}
        route_dist = {label:get_route_length(route[label],data["new_dists"])}
        route_duration = {label:get_route_length(route[label],data["new_times"])}
        route_ferries = {label:get_route_length(route[label],data["new_ferry_costs"])}
        

        return route, route_duration, route_dist, route_ferries
    
def add_wait_times(array, wait_counts, wait_time):
    """wait times to distance matrix are applied column wise"""
    wait_counts = [i*wait_time for i in wait_counts]
    array = np.apply_along_axis(lambda q:q+wait_counts, 1, array)
    return array

def sort_B_by_A(listA,listB):
    listA,listB = zip(*sorted(zip(listA, listB)))
    return listA, listB
      
def get_specific_nodes(array,new_nodes,old_node_to_int):
    new_nodes_to_int = [old_node_to_int[j] for j in new_nodes]
    #https://stackoverflow.com/questions/2732994/python-sort-a-list-and-change-another-one-consequently
    new_nodes_to_int, new_nodes = sort_B_by_A(new_nodes_to_int, new_nodes)
    array = array[np.ix_(new_nodes_to_int,new_nodes_to_int)]
    new_node_to_int_map = dict(zip(new_nodes,[j for j in range(len(new_nodes))]))
    new_int_to_node_map = dict(zip([j for j in range(len(new_nodes))],new_nodes))
    return array, new_node_to_int_map,new_int_to_node_map 

def get_data(df_to_get,isle,num_vehicles,
             cat,depot_node, veh_cap):
    data = {}
    
    # THIS ASSUMES DF IS GROUPED BY NODES
    nodes = list(df_to_get.index)
    if depot_node not in nodes:
        G = graphs[isle]["graph_osgb"]
        x = nx.single_source_shortest_path_length(G,source=depot_node)
        for i in x.keys():
            if i in nodes:
                depot_node = i
                break

    waste_wait_times = {'brewery': 600,
                        'distillery': 600,
                        'fish_farm': 600,
                        'food_processing': 600,
                        'food_waste': 30,
                        'land_farm': 300}
    
    # fornightly res IN KILOGRAMS
    local_res = (df_to_get.groupby("nearest_node").res_biweekly_kg.sum()).round()
    data["new_costs"], data["node_to_int"], data["int_to_node"] = get_specific_nodes(m["cost"][isle],nodes,
                                                                                     graphs[isle]["nodes_to_int_map"])
    
    # add collection wait times (need to get the node to int map first)
    # get the count of properties for each nearest node to add the collection wait time
    nearest_node_property_count = df[df.nearest_node.isin(nodes)].nearest_node.value_counts()
    nearest_node_property_count  = nearest_node_property_count.loc[nearest_node_property_count.index.sort_values()]
    #just to be safe the nodes are in the right order
    nearest_node_property_count = nearest_node_property_count.loc[data["node_to_int"].keys()]
    wait_time = waste_wait_times[cat]
    
    
    data["new_times"], _ , _ = get_specific_nodes(m["time_s"][isle],nodes,
                                                  graphs[isle]["nodes_to_int_map"])
    data["new_times"] = add_wait_times(data["new_times"],
                                       nearest_node_property_count, wait_time)

    
    data["new_ferry_costs"], _,_ = get_specific_nodes(m["cost_ferry"][isle],nodes,
                                                  graphs[isle]["nodes_to_int_map"])
    data["new_dists"], _,_ = get_specific_nodes(m["length_ferry"][isle],nodes,
                                                  graphs[isle]["nodes_to_int_map"])
    
    
    # This sorts the demand so the node order matches the matrix
    sorted_nodes = list(sort_B_by_A([data["node_to_int"][j] for j in local_res.index],local_res.index)[1])
    data["demands"] = list(local_res.loc[sorted_nodes].values)
    
    # OPTIMISING ONLY FOR TIME NOW
    data["distance_matrix"] = data["new_times"]
    
    # Do more vehicles than needed at first - this will be number of 
    #   vehicle days per two weeks
    
    
    cap = veh_cap
    data["num_vehicles"] = num_vehicles
    data["vehicle_capacities"] = [cap] * data["num_vehicles"] 
    
    # The depot comes from the LDP best fit
    data["depot"] = data["node_to_int"][depot_node]
    
    return data

def single_trip_coll(data, int_label=False):
    data_out = deepcopy(data)
    veh_cap = data_out["vehicle_capacities"][0]
    durations = data_out["new_times"]
    dists = data_out["new_dists"]
    ferries = data_out["new_ferry_costs"]
    
    single_trips = {}
    single_dists = {}
    single_durations = {}
    single_ferries = {}
    
    
    for ind,int_load in enumerate(data_out["demands"]):
        data_out["demands"][ind] = int_load % veh_cap
        n_loads = int(int_load // veh_cap)
        
        if not int_label: # calling this on dbscan clusters makes the label output zero, need to correct
            label = ind
        else:
            label = int_label
        if n_loads > 0:
            new = pd.Series(0, index = [label for i in range(n_loads)])
            single_trips[label] = pd.Series([veh_cap]*n_loads,
                                         index = [label for i in range(n_loads)])
            single_dists[label] = deepcopy(new)
            single_durations[label] = deepcopy(new)
            single_ferries[label] = deepcopy(new)
            for n in range(n_loads):
                single_dists[label].iat[n] = 2 * dists[ind,data_out["depot"]]
                single_ferries[label].iat[n] = 2 * ferries[ind,data_out["depot"]]
                single_durations[label].iat[n] = 2 * durations[ind,data_out["depot"]]
    
    # NEED TO UPDATE THE NUMBER OF VEHICLES TOO
    n_veh = bin_packing_solver(data_out["demands"],veh_cap)[0]
    if n_veh!=n_veh:
        n_veh = np.ceil(sum(data_out["demands"])/veh_cap)+1
    n_veh = int(n_veh)
    
    data_out["num_vehicles"] = n_veh
    data_out["vehicle_capacities"] = [data_out["vehicle_capacities"][0]]*n_veh
    return data_out,single_trips, single_durations, single_dists, single_ferries

def bin_packing_solver(values,bin_limit,time_limit = 10):
    """data input of the weights and bin capacity
    https://developers.google.com/optimization/bin/bin_packing"""
    data = {}
    data["weights"] = values
    # if there are any oversized, add the total on at the end
    extra_trips = 0
    for i,val in enumerate(values):
        extra_trips += val // bin_limit
        values[i] = val % bin_limit
        
    data["items"] = list(range(len(values)))
    data['bins'] = data['items']
    data['bin_capacity'] = bin_limit
    
    # Create the mip solver with the SCIP backend.
    solver = pywraplp.Solver.CreateSolver('SCIP')
    solver.SetTimeLimit(time_limit)
    
    # Variables
    # x[i, j] = 1 if item i is packed in bin j.
    x = {}
    for i in data['items']:
        for j in data['bins']:
            x[(i, j)] = solver.IntVar(0, 1, 'x_%i_%i' % (i, j))

    # y[j] = 1 if bin j is used.
    y = {}
    for j in data['bins']:
        y[j] = solver.IntVar(0, 1, 'y[%i]' % j)
    
    # Constraints
    # Each item must be in exactly one bin.
    for i in data['items']:
        solver.Add(sum(x[i, j] for j in data['bins']) == 1)
    
    # The amount packed in each bin cannot exceed its capacity.
    for j in data['bins']:
        solver.Add(
            sum(x[(i, j)] * data['weights'][i] for i in data['items']) <= y[j] *
            data['bin_capacity'])
    
    # Objective: minimize the number of bins used.
    solver.Minimize(solver.Sum([y[j] for j in data['bins']]))
    
    status = solver.Solve()
    
    bin_items_list = []
    bin_weights = []
     
    if status == pywraplp.Solver.OPTIMAL:
        num_bins = 0.
        for j in data['bins']:
            if y[j].solution_value() == 1:
                bin_items = []
                bin_weight = 0
                for i in data['items']:
                    if x[i, j].solution_value() > 0:
                        bin_items.append(i)
                        bin_weight += data['weights'][i]
                if bin_weight > 0:
                    num_bins += 1
                bin_items_list.append(bin_items)
                bin_weights.append(bin_weight)
        num_bins += extra_trips
        return num_bins, bin_items_list, bin_weights

    else:
        print('The BINPACKING problem does not have an optimal solution.')
        return 3*[np.nan]
    
    
#%% RECURSIVE DBSCAN CLUSTERING
#https://www.researchgate.net/publication/329464846_Solving_High_Volume_Capacitated_Vehicle_Routing_Problem_with_Time_Windows_using_Recursive-DBSCAN_clustering_algorithm

def update_clusters(old_list,new_list,update_val):
    _len = len(old_list)
    # assert len(new_list) == sum([i==update_val for i in old_list])
    out = []
    count = 0
    for old in old_list:
        if old==update_val:
            out.append(new_list[count])
            count+=1
        else:
            out.append(old)
    assert len(out) == _len
    return out

def opt_dbscan(df,max_rad,n):
    """ this finds the radius/epsilon that makes the most DBSCAN clusters"""
    most_clusters = 0
    max_rad_range = np.linspace(n,max_rad,round(max_rad/n))
    best_labels = [0]*len(df)
    best_rad = np.nan
    for rad in max_rad_range:
        n_clusters, labels, _ = dbscan(df,rad)
        if n_clusters > most_clusters:
            most_clusters = n_clusters
            best_labels = labels
            best_rad = rad
    return best_labels, best_rad
    

def recursive_dbscan(df, radius_max, n, 
                        max_cluster_res): 
    """ All this needs to do is do a first group (max clusters) then- 
    for the oversized remaining ones also get max clusters"""
    best_clusters, int_rad = opt_dbscan(df,radius_max,n)
    grouped_res = df.groupby(best_clusters).res_biweekly_kg.sum()
    if grouped_res.max()<max_cluster_res:
        return best_clusters
    best_cluster_list = list(set(best_clusters))
    for cluster in best_cluster_list:
        if cluster >= 0: #noise is negative one
            cluster_res = (df.groupby(best_clusters).res_biweekly_kg.sum())[cluster]
            if cluster_res>max_cluster_res:
                cluster_mask = [i==cluster for i in best_clusters]
                n_new = n/10
                # radius_max_new = int_rad
                new_df = df[cluster_mask]
                
                clusters_new, best_rad = opt_dbscan(new_df,radius_max,n_new)
                if best_rad != best_rad:
                    print("dbscan cluster {} did not improve".format(cluster))
                clusters_new = [i+max(best_clusters)+1 if i>0 and i in best_clusters else i for i in clusters_new]
                clusters_new = [i if i!=0 else cluster for i in clusters_new]
                best_clusters = update_clusters(best_clusters,clusters_new,cluster)
    return best_clusters
 
def dbscan(df,epsilon):

    # https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html
    X=np.c_[df.easting,df.northing]
    db=DBSCAN(eps=epsilon, algorithm= "ball_tree", metric= "euclidean").fit(X)
    #get results
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    return n_clusters_,labels,db   



#%%

coll_area = "Islay and Jura"
depot_node = 3693133348

df = pd.read_csv("biomass_resource_df.csv",
                 index_col = 0, low_memory=False)
df = df.loc[df.coll_area==coll_area]

GRAPH = "islay_jura_osmnx_graphs.pickle"
graphs = dict(zip([coll_area],[pd.read_pickle(GRAPH)]))

max_day_length = 8*3600 # max working day of 8 hours

m = pd.read_pickle("dist_matrices_by_metric.pickle")

#%% solve optimised routes

def get_collection_data(df, area_type, waste_cats = [], areas = []):
    """this assumes the df input is already sorted by area and main_type"""

    output = {}
    errors = []
    assert area_type in ['coll_area', 'industrial_group_ref', 'group_road_connected']

    veh_capacities = {"liquid":18000,
           "solid":2500}
    
    veh_types = {'brewery': "liquid",
     'distillery': "liquid",
     'fish_farm': "solid",
     'food_processing': "liquid",
     'food_waste': "solid",
     'land_farm': "solid"}
    
    #Adds these as an input
    if len(waste_cats)==0:
        waste_cats = list(veh_types.keys())
        
    if len(areas)==0:
        areas = df[area_type].unique()
    
    routes = {"routes":{},
          "durations":{},
          "dists":{},
          "ferry_costs":{}}
    
    single = {"routes":{},
              "durations":{},
              "dists":{},
              "ferry_costs":{}}
    
    cluster = {"routes":{},
              "durations":{},
              "dists":{},
              "ferry_costs":{}}
    
    final = {"routes":{},
              "durations":{},
              "dists":{},
              "ferry_costs":{}}
    
    for i in routes:
        for area in df[area_type].unique():
            routes[i][area] = {}
            single[i][area] = {}
            cluster[i][area] = {}
            final[i][area] = {}
    
    time_log = pd.DataFrame(0,index = [i for i in m["time_s"]], 
                            columns = ["cvrp_time","total_time"])
    
    max_rad = {"Islay and Jura": 400
               }
    
    over_sized = max_rad.keys()
    
    cvrp_time = {"Islay and Jura":	40
                 }
    
    # use this for farms as the average amount needs to not be the rounded value
    for cat in waste_cats:
    # for cat in ["food_processing"]:
        output[cat] = pd.DataFrame(0,index =df[area_type].unique(), columns = [])
        veh_cap = veh_capacities[veh_types[cat]]
        for area in  areas:

            print("STARTING {0} FOR {1}".format(cat, area))
            
            new_df = df[(df[area_type]==area)&(df.main_type==cat)]
            
            if len(new_df) == 0:
                print("{0} has no {1} waste\n".format(area,cat))
                continue
                
            # then cluster by nearest node
            x = new_df.groupby("nearest_node").res_biweekly_kg.sum()
            new_df = new_df.groupby("nearest_node").first()
            new_df["nearest_node"] = new_df.index
            new_df.index.name = "new_index"
            new_df["res_biweekly_kg"] = x.values
                     
            #still need coll_area for get_data
            coll_area = df[df[area_type]==area].coll_area.iat[0]
            
            start = time()
            
            # Set up df specific for the island
                    
            for i in final:
                routes[i][area][cat] = {}
                single[i][area][cat] = {}
                cluster[i][area][cat] = {}
                final[i][area][cat] = {}
             
            rr = routes["routes"][area]
            rd = routes["durations"][area]
            rc = routes["dists"][area]
            rf = routes["ferry_costs"][area]
            
            sr = single["routes"][area]
            sd = single["durations"][area]
            sc = single["dists"][area]
            sf = single["ferry_costs"][area]
            
            cr = cluster["routes"][area]
            cd = cluster["durations"][area]
            cc = cluster["dists"][area]
            cf = cluster["ferry_costs"][area]
            
            fr = final["routes"][area]
            fd = final["durations"][area]
            fc = final["dists"][area]
            ff = final["ferry_costs"][area]
            
            # exact num vehicles
            num_vehicles = int(np.ceil(new_df.res_biweekly_kg.sum()/(veh_cap)))
        
            #DO NOT USE THIS FOR DISTANCE CALCS
            main_data = get_data(new_df,coll_area,0,
                                 cat,depot_node,veh_cap)
            
            dbscan_check = (area in over_sized) and (cat=="food_waste") and (len(new_df)>500)
            
            # if the initial df is too big for CVRP, make clustered df
            if dbscan_check:
                
                cluster_lim = 0.5 #THIS IS IN TONNES
                new_labels = recursive_dbscan(new_df,max_rad[coll_area],100,cluster_lim)
                
                #This keeps all the noise points as df entries
                noise_labels = [i==-1 for i in new_labels]
                labels = [i>=0 for i in new_labels]
                noise_df = new_df.loc[noise_labels,:]
                noise_df.loc[:,"labels"] = -1
                noise_df.index = noise_df.nearest_node.values
                               
                #This groups all the labelled points as new points 
                #   (assumes the first point is the coordinate of the new collection)                                                       
                db_df = new_df.loc[labels,:]
                db_df.loc[:,"labels"] = [label for label,check in zip(new_labels,labels) if check]
                x = db_df.groupby(db_df.labels).res_biweekly_kg.sum()
                db_df = db_df.groupby(db_df.labels).first()  
                db_df["labels"] = db_df.index
                db_df.index = db_df.nearest_node.values
                db_df.loc[:,"res_biweekly_kg"] = x.values
                
                
                grouped_df = pd.concat([db_df,noise_df])
                
            else:
                grouped_df = new_df
                

            # get data for CVRP, clustered or not
            data = get_data(grouped_df,coll_area,num_vehicles,
                            cat,depot_node,veh_cap)
            
            # for some cases there will be a single node which is also the depot-
            # COSTS ARE ZERO
            if data["distance_matrix"].shape[0]==1:
                continue
    
            #do single trips for excess nodes (>veh_cap)
            data, sr[cat], sd[cat], sc[cat], sf[cat] = single_trip_coll(data)
            
            # do CVRP for groupded nodes without excess nodes
            # need to recalculate number of vehicles with single trips taken out
            num_vehicles = int(np.ceil(sum(data["demands"])/
                                       (veh_cap)))
            data["num_vehicles"] = num_vehicles
            data["vehicle_capacities"] = [veh_cap] * data["num_vehicles"]
            data["demands"][data["depot"]] = 0
            
            ## BELOW WITH TIME LIMIT IS FOR GUIDED LOCAL SEARCH - WONT END WITHOUT TIMER
            start_cvrp = time()
            if cat == "food_waste" and len(data["demands"])>500:
                t = cvrp_time[coll_area]
            else:
                t = 1
            rr[cat], rd[cat], rc[cat], rf[cat] = cvrp_solver(data,area,t)
            
            if rr[cat] != rr[cat]:
                # for some reason depot demand>0 fails so
                data["demands"][data["depot"]] = 0
                data["num_vehicles"] = num_vehicles+2
                data["vehicle_capacities"] = [veh_cap] * data["num_vehicles"]
                t = 50*t
                if t>2000:
                    t = 2000
                rr[cat], rd[cat], rc[cat], rf[cat] = cvrp_solver(data,area,t)
                if rr[cat] != rr[cat]:
                    errors.append([cat,area,t,data["num_vehicles"]])
                    continue
                    
            time_log.at[area,"cvrp_time"] = time()-start_cvrp
            # print("initial CVRP solver completed in ", time()-start_cvrp)
            
            # this is the basis of the final routes, just need to add in single/cluster trips
            fr[cat] = copy(rr[cat])
            fd[cat] = copy(rd[cat])
            fc[cat] = copy(rc[cat])
            ff[cat] = copy(rf[cat])
                    
            # now need to calculate the costs of individual clusters
            if dbscan_check:
                
                for label in set(new_labels):
                    if label>=0:
                        label_mask = [i==label for i in new_labels]
                        sub_df = new_df.loc[label_mask]
                        
                        num_vehicles = int(np.ceil(sub_df.res_biweekly_kg.sum()/
                                       (veh_cap)))
                        subdata = get_data(sub_df,coll_area,num_vehicles,
                                           cat, depot_node,veh_cap)

                        #set the node nearest the actual depot as the starting/end point
                        nodes = sub_df.nearest_node.unique()
                        nodes_int = [main_data["node_to_int"][i] for i in nodes]
                        closest_depot = nodes[pd.Series(main_data["new_times"][main_data["depot"],nodes_int]).idxmin()]
                        subdata["depot"] = subdata["node_to_int"][closest_depot]
                        
                        if num_vehicles==1:
                            cr[cat][label], cd[cat][label], cc[cat][label], cf[cat][label] = salesperson_solver_no_return(subdata,label)
                        else:
                            # CVRP cant solve if any node > veh_cap
                            if any(sub_df.res_biweekly_kg >= veh_cap):
                                ind = list(grouped_df.labels).index(label)
                                subdata, x, y, z, z2 = single_trip_coll(subdata, label)
                                sr[cat][label], sd[cat][label], sc[cat][label], sf[cat][label] = x[label], y[label], z[label], z2[label]
                            num_vehicles = int(np.ceil(sub_df.res_biweekly_kg.sum()/
                                       (veh_cap)))
                            cr[cat][label], cd[cat][label], cc[cat][label], cf[cat][label] = cvrp_solver(subdata,area,5)
                        
                        # recombine costs of clustered trips with the optimised routes for total
                        if len(sr[cat])>0:
                            for single_trip in sr[cat]:
                                if new_labels[single_trip] == label:
                                    ind = max(rr[cat].keys())+1
                                    fr[cat][ind] = sr[cat][single_trip].iloc[0]
                                    fd[ind] = sd[cat][single_trip].iloc[0] + sum(cd[cat][label].values())
                                    fc[cat][ind] = sc[cat][single_trip].iloc[0] + sum(cc[cat][label].values())
                                    ff[cat][ind] = sf[cat][single_trip].iloc[0] + sum(cf[cat][label].values())
                                    ind += 1
                                          
                    # This adds single trips to noise labels to the final results
                    elif label == -1:
                        if len(sr[cat])>0:
                            for single_trip in sr[cat]:
                                if new_labels[single_trip] == -1:
                                    ind = max(rr[cat].keys())+1
                                    for i in sr[cat]:
                                        for j in range(len(sr[cat][i])):
                                            fr[cat][ind] = sr[cat][i].iloc[j]
                                            fd[cat][ind] = sd[cat][i].iloc[j]
                                            fc[cat][ind] = sc[cat][i].iloc[j]
                                            ff[cat][ind] = sf[cat][i].iloc[j]
                                            ind += 1 
                # this adds cluster trip costs to the routes
                noise_check = int(any([i==-1 for i in new_labels]))
                cluster_added = dict(zip([i for i in set(new_labels) if 1>=0],[False]*(len(set(new_labels))-noise_check)))
                for route_ind in rr[cat]:
                    for grouped_df_node in rr[cat][route_ind].index:
                        label = new_labels[grouped_df_node]
                        if label >= 0 and not cluster_added[label]:
                            # print(label)
                            fd[cat][route_ind] += sum(cd[cat][label].values())
                            fc[cat][route_ind] += sum(cc[cat][label].values())
                            ff[cat][route_ind] += sum(cf[cat][label].values())
                            cluster_added[label] = True
                            
            else:
                # only run if solution exists
                if rr[cat] == rr[cat]:
                    if len(sr[cat])>0:
                        ind = max(rr[cat].keys())+1
                        for i in sr[cat]:
                            for j in range(len(sr[cat][i])):
                                fr[cat][ind] = sr[cat][i].iloc[j]
                                fd[cat][ind] = sd[cat][i].iloc[j]
                                fc[cat][ind] = sc[cat][i].iloc[j]
                                ff[cat][ind] = sf[cat][i].iloc[j]
                                ind += 1
                    
            time_taken = time()-start
            time_log.at[area,"total_time"] = time_taken
            print(("{0} took {1} seconds\n").format(area,time_taken))
            
            # ADD RESULTS TO OUTPUT
            output[cat].at[area,"total_time"] = sum(final["durations"][area][cat].values())
            output[cat].at[area,"num_trips"] = len(final["durations"][area][cat])
            output[cat].at[area,"total_distance"] = sum(final["dists"][area][cat].values())/1000
            output[cat].at[area,"annual_dist"] = output[cat].at[area,"total_distance"] * 26
            output[cat].at[area,"ferry_costs"] = sum(final["ferry_costs"][area][cat].values())
            
            durations = list(final["durations"][area][cat].values())
            if max(durations)>max_day_length:
                max_day = max(durations)
            else:
                max_day = max_day_length
            
            x = bin_packing_solver(durations,max_day,1)[0]
            if x!=x:
                output[cat].at[area,"num_coll_days"] = np.ceil(output[cat].at[area,"total_time"] / max_day)
            else:
                output[cat].at[area,"num_coll_days"] = x
            
    return output,errors

#%%

def res_scen_df(df,scen):
    out = deepcopy(df)
    out["res_annual_scen"] = out["res_annual_scen"].apply(lambda q: literal_eval(q)[scen]).astype(float)
    out["res_annual_scen"].replace(0,np.nan,inplace=True)
    out.dropna(subset="res_annual_scen",inplace=True)
    out["res_biweekly"] = out["res_annual_scen"]/26
    out["res_biweekly_kg"] = (out["res_biweekly"]*1000).round()
    out["biogas_energy"] = out["res_annual_scen"] * out["biogas_yield"]
    out = out[out.res_biweekly>0]
    return out

FINAL = "2. FINAL COLLECTON COSTS FOR SCENARIOS"

try:
    coll_costs = pd.read_pickle(FINAL)
except:
    coll_costs = {}
    errors = {}
    for res_scen in ["high","mean","low"]:
        if res_scen not in coll_costs:
            coll_costs[res_scen] = {}
            errors[res_scen] = {}
        new_df = res_scen_df(df,res_scen)
        # for area in ["industrial_group_ref","coll_area","group_road_connected"]:
        for area in ["industrial_group_ref"]:
            if area not in coll_costs[res_scen]:
                print("\n\nSTARTING RES SCENARIO: {0}\nAREA TYPE: {1}\n\n\n".format(res_scen,area))
                coll_costs[res_scen][area], errors[res_scen][area] = get_collection_data(new_df,area)
                
            pd.to_pickle(coll_costs,FINAL)


#%% SORT RESULTS

out = []

for res_scen in coll_costs:
    new1 = []
    for fac_scen in coll_costs[res_scen]:
        new2 = []
        for main_type in coll_costs[res_scen][fac_scen]:
            coll_costs[res_scen][fac_scen][main_type]["main_type"] = main_type
            coll_costs[res_scen][fac_scen][main_type]["fac_scen"] = fac_scen
            coll_costs[res_scen][fac_scen][main_type]["res_scen"] = res_scen
            new2.append(coll_costs[res_scen][fac_scen][main_type])
        new1.append(pd.concat(new2))
    out.append(pd.concat(new1))
    
colldf = pd.concat(out)
colldf["annual_ferry_cost"] = colldf.ferry_costs*26
colldf["total_cost"] = colldf.annual_ferry_cost+colldf.annual_dist*1.25
colldf.reset_index(inplace=True)
x = list(colldf.columns)
x[0] = "ref"
colldf.columns = x

x = pd.DataFrame(0,columns = ["high","mean","low"], index = df.index)
y = pd.DataFrame(0,columns = ["high","mean","low"], index = df.index)
for i in x:
    x[i] = res_scen_df(df,i)["res_biweekly_kg"]
    y[i] = res_scen_df(df,i)["biogas_energy"]
    
x.fillna(0,inplace=True)
y.fillna(0,inplace=True)
colldf[["res_biweekly_kg","biogas_energy"]] = 0

for i in colldf.index:
    ref = colldf.at[i,"ref"]
    res_scen = colldf.at[i,"res_scen"]
    main_type = colldf.at[i,"main_type"]
    fac_scen = colldf.at[i,"fac_scen"]
    res = df[(df.main_type==main_type)&(df[fac_scen]==ref)].index
    if len(res)>0:
        colldf.at[i,"res_biweekly_kg"] += x.loc[res,res_scen].sum()/3
        colldf.at[i,"biogas_energy"] += y.loc[res,res_scen].sum()/3

colldf["res_biweekly"] = colldf["res_biweekly_kg"] / 1000       
colldf["res_annual"] = colldf["res_biweekly"] * 26 

colldf["cost_per_tonne"] = colldf.total_cost/colldf.res_annual

veh_types = {'brewery': "liquid",
     'distillery': "liquid",
     'fish_farm': "solid",
     'food_processing': "liquid",
     'food_waste': "solid",
     'land_farm': "solid"}

colldf["type"] = colldf.main_type.apply(lambda q: veh_types[q])

for i in colldf.index:
    ref = colldf.at[i,"ref"]
    fac_scen = colldf.at[i,"fac_scen"]
    la = df[df[fac_scen]==ref]["la"].iat[0]
    colldf.at[i,"la"] = la

veh_type = {'brewery':"liquid",
 'distillery':"liquid",
 'food_processing':"liquid",
 'food_waste':"food",
 'land_farm':"animal",
 'fish_farm':"animal"}
colldf["veh_type"] = colldf.main_type.apply(lambda q: veh_type[q])


colldf.to_csv("transport_costs.csv")

#%% Get number of vehicles per la per veh type

veh_per_la = pd.DataFrame(columns = ["res_scen","fac_scen","veh_type","la",
                                     "num_days","annual_dist",
                                     "annual_ferry_cost","num_vehs","total_time"],
                          index = [i for i in range(162)])
colldf.dropna(inplace=True)
i = 0
for res_scen in colldf.res_scen.unique():
    for fac_scen in colldf.fac_scen.unique():
        for veh in colldf.veh_type.unique():
            for la in colldf.la.unique():
                veh_per_la.loc[i] = [res_scen, fac_scen, veh, la, 0, 0, 0, 0, 0]
                
                j = colldf[(colldf.res_scen==res_scen)&(colldf.fac_scen==fac_scen)&(colldf.veh_type==veh)&(colldf.la==la)]
                if len(j)>0:
                    if len(j) == 1:
                        veh_per_la.at[i,"num_days"] = j["num_coll_days"].iat[0]
                    else:
                        veh_per_la.at[i,"num_days"] = bin_packing_solver(list(j.total_time.values),
                                                                       28800)[0]
                        if veh_per_la.at[i,"num_days"] != veh_per_la.at[i,"num_days"]:
                            veh_per_la.at[i,"num_days"] = (sum(j.total_time.values) // 28800) + 1
                    
                    #round number of days for isolated LAs
                    veh_per_la.at[i,"num_vehs"] = veh_per_la.at[i,"num_days"]/10
                    if la in ["Shetland","Orkney","Na-h Eileanan Siar"]:
                        veh_per_la.at[i,"num_vehs"] = np.ceil(veh_per_la.at[i,"num_days"])
                        
                    veh_per_la.at[i,"annual_dist"] = j.annual_dist.sum()
                    veh_per_la.at[i,"annual_ferry_cost"] = j.annual_ferry_cost.sum()
                    veh_per_la.at[i,"total_time"] = j.total_time.sum()
                i += 1
                
veh_per_la["num_vehs"] = veh_per_la[["la","num_days"]].apply(lambda q: np.ceil(q.iat[1]/10) if q.iat[0] in ["Shetland","Orkney","Na h-Eileanan Siar"] else q.iat[1]/10, axis=1)

veh_per_la.to_csv("vehicle_numbers.csv")

    