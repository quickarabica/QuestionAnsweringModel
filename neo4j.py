from py2neo import Graph
import graphviz

# Connect to Neo4j
graph = Graph("bolt://localhost:7687", auth=("neo4j", "12345678"))

# Run Cypher query
query = """MATCH (e {name: 'whole farm planning toolkits'})-[r:Coreference]-(related)
RETURN e,r,related"""
results = graph.run(query).data()

# Visualize and generate graph
dot = graphviz.Digraph(comment='Neo4j Graph')

for result in results:
    # Extract node names and relationship details
    node1_name = result['e']['name']  # Extract the 'name' property from node 'e'
    node2_name = result['related']['name']  # Extract the 'name' property from 'related'
    relationship = result['r'].__class__.__name__  # Get the relationship type correctly

    # Create nodes and edge for the graph
    dot.node(node1_name, node1_name)  # Create the first node
    dot.node(node2_name, node2_name)  # Create the second node
    dot.edge(node1_name, node2_name, label=relationship)  # Create the edge with relationship label

# Save as PNG
dot.render('neo4j_graph', format='png')
