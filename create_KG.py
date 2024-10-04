import pandas as pd
from neo4j import GraphDatabase

# Load the dataset
data = pd.read_csv(r"C:\Users\priya\Desktop\nlp2\agri.csv")

# Connect to the Neo4j database
uri = "bolt://localhost:7687"
driver = GraphDatabase.driver(uri, auth=("neo4j", "12345678"))  # Replace with your Neo4j credentials

def create_entity(tx, entity, entity_type):
    query = f"""
    MERGE (e:{entity_type} {{name: $entity}})
    """
    tx.run(query, entity=entity)

def create_relationship(tx, head_entity, head_entity_type, relation, tail_entity, tail_entity_type):
    query = f"""
    MATCH (head:{head_entity_type} {{name: $head_entity}})
    MATCH (tail:{tail_entity_type} {{name: $tail_entity}})
    MERGE (head)-[r:{relation}]->(tail)
    """
    tx.run(query, head_entity=head_entity, tail_entity=tail_entity)

with driver.session() as session:
    for _, row in data.iterrows():
        session.execute_write(create_entity, row['entity1'], row['head_label'])
        session.execute_write(create_entity, row['entity2'], row['tail_label'])
        session.execute_write(create_relationship, row['entity1'], row['head_label'], row['relation'], row['entity2'], row['tail_label'])

driver.close()
