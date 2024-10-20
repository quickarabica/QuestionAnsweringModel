from pyvis.network import Network
from py2neo import Graph

graph = Graph("bolt://localhost:7687", auth=("neo4j", "12345678")) 
def visualize_graph(cypher_query):
    results = graph.run(cypher_query).data()
    net = Network(notebook=True, cdn_resources='remote')
    
    # Create a Network object for the interactive graph
    net = Network(notebook=True)

    # Add nodes and edges to the interactive graph
    for result in results:
        node1_name = result['e']['name']
        node2_name = result['related']['name']
        relationship = result['r'].__class__.__name__

        # Add nodes with a unique ID
        net.add_node(node1_name, label=node1_name)
        net.add_node(node2_name, label=node2_name)

        # Add an edge with relationship label
        net.add_edge(
            node1_name,
            node2_name,
            title=relationship, 
            label=relationship,  # Display the relationship name on the edge
            color='red', 
            arrows='to', 
            length=220  # You can adjust the length and color as needed
        )

    # Save the interactive graph as an HTML file
    net.save_graph('answer.html')
    print("Interactive graph generated and saved as 'neo4j_graph.html'.")

visualize_graph("MATCH (e {name: 'Farmers'})-[r:Helps_In]->(related) RETURN e,r,related")