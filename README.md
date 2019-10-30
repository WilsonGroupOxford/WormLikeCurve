**ring_finder**
This is a python code designed to identify the rings in a network structure. It requires the connections and positions of each node in the graph, and constructs a Delauney triangulation in which the rings are well defined by their simplicies.

By removing simplex edges that do not exist in the original graph, regions are formed. Each region belongs to a different ring.
