
from neo4j import GraphDatabase
from kgbuilder import KGBuilder
from refined.data_types.base_types import Span


URI = "neo4j://localhost"
AUTH = ("neo4j","prueba_1234")


def get_entity_from_dicc(dictionary:dict,term:str):
    entity = dictionary.get(term)
    if entity is None:
        for key in dictionary:
            if key in term or term in key:
                entity = dictionary.get(key)
                break
    
    return entity


def get_entity_id(entity):
    if entity.predicted_entity is None:
        name = None
    else: 
        name =  entity.predicted_entity.wikidata_entity_id  

    return name


def hash_of_span(self)-> int:
    
    text = self.text if self.text is not None else "None"
    entity_id = "ENTITY NONE"
    wikipedia_entity_title = "ENTITY NONE"
    
    if self.predicted_entity is not  None:    
        entity_id = self.predicted_entity.wikidata_entity_id if self.predicted_entity.wikidata_entity_id is not None else "None"
        wikipedia_entity_title = self.predicted_entity.wikipedia_entity_title if self.predicted_entity.wikipedia_entity_title is not None else "None"
    
    
    return hash(text + " " + entity_id + " "+ wikipedia_entity_title)

Span.__hash__ = hash_of_span
#builder = KGBuilder()
#spans,triplets,spans_el_base,spans_re,triplets_base,triplets_base_er = builder.build_graph("Toyota Motor Corporation, founded in 1937 by Kiichiro Toyoda, is one of the world's leading automotive manufacturers. The company originated from the Toyoda Automatic Loom Works, which diversified into automobile production under Kiichiro's vision. Toyota's first passenger car, the Model AA, was produced in 1936. Post-World War II, the company faced financial difficulties but rebounded with innovative manufacturing techniques, including Just-In-Time production, which revolutionized the industry. The introduction of the Corolla in 1966 cemented Toyota's reputation for reliability and affordability. In the 21st century, Toyota became a pioneer in hybrid technology with the launch of the Prius in 1997, leading the global shift towards sustainable automotive solutions. Today, Toyota continues to innovate with advancements in electric vehicles, hydrogen fuel cells, and autonomous driving technologies, maintaining its position as a global automotive leader.")
builder = KGBuilder()
def method_3(text):
    spans,triplets,_,_,_,_ = builder.build_graph(text)
    entities = set()
    relations_set = set()
    relationships_list = []
    spans_list = [] 
    for span in spans:
        id = get_entity_id(span)
        if id is not None:
            entities.add(id)
            spans_list.append(span)
    
    for triplet in triplets:
        subject = get_entity_id(triplet['head'])
        predicate = triplet['type']
        obj = get_entity_id(triplet['tail'])
        if subject is not None and obj is not None:
            relation_text = subject+predicate+obj
            if relation_text not in relations_set:
                relations_set.add(relation_text)
                relation_dicc = {
                            "subject":{
                                "uri":subject
                            },
                            "predicate":{
                                "surfaceform":predicate
                            },
                            "object":{
                                "uri":obj
                            }
                        }
                relationships_list.append(relation_dicc)

    
    return relationships_list,list(entities),spans_list


def extract_chunks(phrase,tokenizer):

    tokens = tokenizer.tokenize(phrase)
    n_tokens = len(tokens)
    chunk_size = n_tokens
    n_chunks = 1
    while chunk_size > 1024:
        chunk_size = chunk_size/2
        n_chunks = n_chunks*2

    chunks = []
    for i in range(0,n_chunks):
        chunks.append(tokens[i*chunk_size:i+1*chunk_size])

    return chunks



if __name__ == '__main__':
    builder = KGBuilder()
    while True:
        phrase = input("Enter a phrase to extract information: ")
        #chunks = extract_chunks(phrase,builder.triplet_extractor.tokenizer)
        relationships, entities,spans = method_3(phrase)
        print(relationships)

        
        
    with GraphDatabase.driver(URI,auth=AUTH) as driver:
        driver.verify_connectivity()
    