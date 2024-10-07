from neo4j import GraphDatabase

# Connect to the Neo4j database
uri = "bolt://localhost:7687"
driver = GraphDatabase.driver(uri, auth=("neo4j", "12345678"))  # Replace with your Neo4j credentials

def get_entity_info( entity, relation=None, related_entity=None):
    if entity and relation and related_entity:
        query = f"""
        MATCH (e {{name: '{entity}'}})-[r:{relation}]->(related)
        RETURN e,r,related
        
        """
        print("Generated Query:")
        print(query)


def fetch_detailed_info(entities, relations=None, related_entities=None):
    detailed_info = []
    with driver.session() as session:
        for entity in entities:
            if relations and related_entities:
                for relation, related_entity in zip(relations, related_entities):
                    # Fetch the relationship information
                    relationships = session.read_transaction(get_entity_info, entity, relation, related_entity)
                    if relationships:  # Append only if data is found
                        detailed_info.append(relationships)
            elif relations:
                for relation in relations:
                    relationships = session.read_transaction(get_entity_info, entity, relation)
                    if relationships:
                        detailed_info.append(relationships)
            else:
                info = session.read_transaction(get_entity_info, entity)
                if info:
                    detailed_info.append(info)
    return detailed_info

# Example test data
test_entities = ["Multi Layer Perceptron"]
test_relations = ["Synonym_Of"]
test_related_entities = ["MLP"]

# Fetching detailed info for the test entities, relations, and related entities
data = fetch_detailed_info(test_entities, test_relations, test_related_entities)

# # Displaying results without printing unwanted output
# if data:
#     for result in data:
#         print("Result: ")
#         print(f" {result['entity1']} {result['head_label']} , {result['relation']} , {result['entity2']} {result['tail_label']}")
# else:
#     print("No results found.")

if data:
    for result in data:
        # Extract 'head_entity' (e node), 'relation' (r relationship), and 'tail_entity' (related node)
        head_entity = result['e']['name']  # Get the 'name' property of the 'e' node
        head_label = list(result['e'].labels)[0]  # Get the first label of the 'e' node
        relation = result['r'].type  # Get the type of the 'r' relationship
        tail_entity = result['related']['name']  # Get the 'name' property of the 'related' node
        tail_label = list(result['related'].labels)[0]  # Get the first label of the 'related' node
        
        # Print the extracted information
        print(f"Head Entity: {head_entity}, Head Label: {head_label}, Relation: {relation}, Tail Entity: {tail_entity}, Tail Label: {tail_label}")
        print("\n")
else:
    print("No results found.")
