
from neo4j import GraphDatabase
import wikidata
from kgbuilder import KGBuilder
from refined.data_types.base_types import Span
import spacy
from transformers import DistilBertTokenizer, DistilBertModel
from wikidata.client import Client
import logging
import json

logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)



URI = "neo4j://localhost"
AUTH = ("neo4j","prueba_1234")

properties_dict = {}


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


def extract_chunks(nlp, phrase,tokenizer,model):
    doc = nlp(phrase)
    chunks = []
    index = 0
    for sent in doc.sents:
        chunk = {
            'index': index,
            'text': sent.text,
            'start': sent.start_char,
            'end': sent.end_char,
            'embedding': generate_contextual_embedding_from_phrase(sent.text,model=model,tokenizer=tokenizer),
            'spans':[]
        }
        chunks.append(chunk)
        index += 1

    return chunks



def generate_contextual_embedding_from_phrase(phrase,model,tokenizer):
    encoded_input = tokenizer(phrase,return_tensors='pt')
    outputs= model(**encoded_input)
    embedding = str(outputs.last_hidden_state.mean(dim=1).tolist()[0])
    
    return embedding


def create_relationships_between_entities(driver,relationships):
    stmt = """

    MATCH (e1:Entity{wikidata_id:$e1_id}),(e2:Entity{wikidata_id:$e2_id})

    """
    
    for rel in relationships:
        head_uri = rel['subject']['uri']
        type_rel = rel['predicate']['surfaceform'].upper().replace(" ",'_')
        tail_uri = rel['object']['uri']
        rel_stmt = stmt +"MERGE (e1)-[:"+type_rel+"]->(e2)"
        params = {'e1_id':head_uri,'e2_id':tail_uri}

        with driver.session() as session:
            session.run(rel_stmt,params)



def link_entity_mentions_with_chunks(driver,chunks):

    stmt = """

    MATCH (c:Chunk)
    WHERE id(c) = $chunk_id
    MATCH (e:Entity{wikidata_id:$wikidata_id})
    CREATE (e)<-[:MENTION{as:$mention,index_start:$start,index_end:$end,confidence:$conf}]-(c)
    
    """

    for chunk in chunks:
        for span in chunk['spans']:
             params = {'wikidata_id':span.predicted_entity.wikidata_entity_id,
                       'chunk_id':chunk['db_id'],
                       'mention':span.text,
                       'conf':span.entity_linking_model_confidence_score,
                       'start':span.start,
                       'end':len(span.text)+span.start-1}
             with driver.session() as session:
                session.run(stmt,params) 


def link_chunks_with_text_in_database(driver,chunks_db_id,text_id):
    stmt = """
        MATCH (t:Text),(c:Chunk)
        WHERE id(t) = $text_id and id(c) = $chunk_id
        CREATE (c)-[:PART_OF]->(t)
    """

    for chunk_id in chunks_db_id:
        params = {'text_id':text_id,'chunk_id':chunk_id}
        with driver.session() as session:
           session.run(stmt,params) 

def create_chunks_in_database(driver,chunks,text_id):
    stmt = "CREATE (c:Chunk {index:$index, text: $text,index_start:$start,index_end:$end,embedding:$embedding}) RETURN id(c) as nodeId"
    chunks_db_id = []
    for chunk in chunks:
        params = {'text':chunk['text'],'start':chunk['start'],'end':chunk['end'],'embedding':chunk['embedding'],'index':chunk['index']}

        with driver.session() as session:
           result = session.run(stmt,params) 
           node_id = result.single()['nodeId']
           chunk['db_id'] = node_id
           chunks_db_id.append(node_id)

    link_chunks_with_text_in_database(driver,chunks_db_id,text_id)

    return chunks_db_id




def create_entities_in_database(driver,entities):

    stmt = "MERGE (c:Entity {wikidata_id:$id})"
    for ent in entities:
        params = {'id':ent}

        with driver.session() as session:
           result = session.run(stmt,params)
    
           

def create_text_in_database(driver,phrase):
    stmt = "CREATE (t:Text {text: $text, createdAt: datetime()}) RETURN id(t) as nodeId"
    params = {"text": phrase}
    
    with driver.session() as session:
        result = session.run(stmt, params)
        node_id = result.single()['nodeId']
        return node_id


def distribute_spans_in_chunks(spans,chunks):
    for span in spans:
        for chunk in chunks:
            if span.start >= chunk['start'] and len(span.text) + span.start - 1 <= chunk['end']:

                if chunk.get('spans'):
                    chunk['spans'].append(span)
                else:
                    chunk['spans'] = [span]

                break



def map_entities_with_external_representation(driver,client):
    stmt = "MATCH (n:Entity) where n.`label` is null return n"

    with driver.session() as session:
        result = session.run(stmt)
        for record in result:
            wikidata_id = record["n"]["wikidata_id"]
            try:
                entity = client.get(wikidata_id,load=True)

                
                description = entity.description.texts['en']
                labels = entity.attributes['labels']
                label_es = None
                label_ca = None
                label = None
                if labels.get('en',None) is not None:
                    label = labels.get('en',None)['value']
                if labels.get('es',None) is not None:
                    label_es = labels.get('es',None)['value']
                if labels.get('ca',None) is not None:
                    label_ca = labels.get('ca',None)['value']
                
                stmt = "MATCH (n:Entity) where n.`wikidata_id`= $wikidata_id SET n.`label` = $label, n.description = $description, n.`label:es`= $label_es, n.`label:ca`= $label_ca"
                params = {'label':label,'description':description,'label_es':label_es,'label_ca':label_ca,'wikidata_id':wikidata_id}

                dict_params = {
                    
                }
                for i in entity.iterlists():
                    for j in i[1]:
                        try:
                            property = None
                            if type(j) == str:
                                property = properties_dict[i[0].id].lower().replace(' ','_')
                                unity =  str(j) 
                                dict_params[property] = j
                            if type(j) == wikidata.quantity.Quantity:
                                property = properties_dict[i[0].id].lower().replace(' ','_')
                                unity =  "," +  str(j.unit.label) if j.unit is not None else ""
                                dict_params[property] = str(j.amount) + unity
                            if type(j) == wikidata.globecoordinate.GlobeCoordinate:
                                property = properties_dict[i[0].id].lower().replace(' ','_')
                                dict_params[property] = "point({latitude: "+str(j.latitude)+",longitude:"+str(j.longitude)+"})"
                        except Exception as e:
                            pass


                stmt_part = ""
                dict_update = {}
                i = 0
                for key in dict_params.keys():
                    parameter = "p"+str(i)
                    stmt_part += ", n.`"+key+"`=$"+parameter
                    dict_update[parameter] = dict_params[key]
                    i += 1

                dict_update.update(params)
                session.run(stmt+stmt_part,dict_update)

            except Exception as e:
                print(e)

            
            

if __name__ == '__main__':
    builder = KGBuilder()
    client = Client()
    nlp = spacy.load("en_core_web_sm")
    
    with open('properties.json','r') as file:
        properties_dict = json.load(file)
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained("distilbert-base-uncased")


    while True:
        phrase = input("Enter a phrase  with no newline to extract information: ")
        print("1. Extracting information from the text")
        chunks = extract_chunks(nlp,phrase,tokenizer,model)
        relationships, entities,spans = method_3(phrase)
        distribute_spans_in_chunks(spans,chunks)

        with GraphDatabase.driver(URI,auth=AUTH) as driver:
            print("2. Storing information in database")
            text_id = create_text_in_database(driver,phrase)
            chunks_id = create_chunks_in_database(driver,chunks,text_id)
            create_entities_in_database(driver,entities)
            link_entity_mentions_with_chunks(driver,chunks)
            create_relationships_between_entities(driver,relationships)
            print("3. Updating entity information")
            map_entities_with_external_representation(driver,client)

