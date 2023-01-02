# Biowaste-to-energy_techno-economic_modelling
Optimisation of waste collection clustering to model scenarios of biowaste-to-energy for the Scottish islands

See paper published address XXXX

There are three scripts to the model (preparation and cleaning of the bioresource and OSMnx graphs has been carried out separately).

1. Distance matrix calculation- for the OSMnx graph, a matrix of distances for all nodes is created.
2. CVRP routing opimisation with recursive DBSCAN clustering- using the distance matrices, acollection routes are optimised. This returns 
      collection costs for each scenario.
3. Techno-economic modelling- for scenarios of collection costs, generation technology and catchment area, the levelised cost of energy 
      (LCOE) is calculated.
