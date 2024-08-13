from transformers import pipeline
from refined.inference.processor import Refined
import tqdm as notebook_tqdm
import torch
from typing import List
from typing import Dict
from typing import Set
from typing import Iterable
from refined.data_types.doc_types import Doc
from refined.data_types.modelling_types import BatchedElementsTns
from refined.utilities.preprocessing_utils import convert_doc_to_tensors
from refined.data_types.base_types import Span
from refined.utilities.preprocessing_utils import pad
from refined.data_types.modelling_types import ModelReturn
from collections import defaultdict
from refined.utilities.general_utils import round_list
from refined.data_types.base_types import Entity
import re
import pandas as pd
import json
from langchain.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser



class KGBuilder:
    # Models that build the data
    # Entity Linking Model
    refined = Refined.from_pretrained(model_name='wikipedia_model_with_numbers', entity_set='wikidata',use_precomputed_descriptions=True)
    # Relation Extraction Model
    triplet_extractor = pipeline('text2text-generation',model='Babelscape/rebel-large',tokenizer='Babelscape/rebel-large',device='cuda')
    # LLM for relation classification
    #llm = ChatOllama(model="llama3:8b")
    
    def __find_all(substring, string):
        return [m.start() for m in re.finditer(re.escape(substring), string)]
    
    def filter_spans(spans:List[Span])->List[Span]:
        spans = [span for span in spans if span.entity_linking_model_confidence_score is not  None and span.entity_linking_model_confidence_score >= 0.75 and span.predicted_entity.wikidata_entity_id is not None]
        i = 0
        while i < len(spans):
            j = i + 1
            while j < len(spans):
                if spans[j].text == spans[i].text and spans[j].predicted_entity.wikidata_entity_id == spans[i].predicted_entity.wikidata_entity_id:
                    del spans[j]

                j += 1
                        
            
            i += 1

        
        return spans
    
    def filter_relationships(triplets:List[Dict])->List[Dict]:
        relations = set()
        for i in triplets:
            relations.add(str(i))

        triplets = []
        for i in relations:
            try:
                triplets.append(json.loads(i.replace("'",'"')))
            except Exception as e:
                pass


        return triplets
    
    def entity_linking(text:str)->List[Span]:
        spans_el = KGBuilder.refined.process_text(text)
        spans_el = KGBuilder.filter_spans(spans_el)
        return spans_el
      
    def relation_extraction(text:str)->List[Dict]:
        triplets = KGBuilder.extract_triplets(KGBuilder.triplet_extractor.tokenizer.batch_decode([KGBuilder.triplet_extractor(text, return_tensors=True, return_text=False)[0]["generated_token_ids"]])[0])
        triplets = KGBuilder.filter_relationships(triplets)

        return triplets
     
    def link_spans_with_relationships(triplets:List[Dict],spans:List[Span]):
        i = 0
        entities_with_relation = set()
        while i < len(triplets):
            for span in spans:
                if triplets[i]['head'] == span.text:
                    triplets[i]['head'] = span
                    entities_with_relation.add(span)
                
                if triplets[i]['tail'] == span.text:
                    triplets[i]['tail'] = span
                    entities_with_relation.add(span)

            if not isinstance(triplets[i]['head'],Span) or not isinstance(triplets[i]['tail'],Span) or triplets[i]['head'].predicted_entity.wikidata_entity_id == triplets[i]['tail'].predicted_entity.wikidata_entity_id:
                del triplets[i]
                i = i - 1
            

            i += 1

        entities_set = set()
        for span in spans:
            entities_set.add(span)

        entities_with_no_relation = entities_set.difference(entities_with_relation)

        return (triplets,entities_with_no_relation)
    
    def extract_triplets_from_spans_with_no_relationship(text:str,spans_no_relationship:List[str])->List[Dict]:
        model = KGBuilder.triplet_extractor.model
        tokenizer = KGBuilder.triplet_extractor.tokenizer
        new_triplets = []
        gen_kwargs = {
            "max_length": 1024,
            "length_penalty": 1,
            "num_beams": 3,
        }

        model_inputs = tokenizer(text, max_length=1024, padding=True, truncation=True, return_tensors = 'pt')

        for span in  spans_no_relationship:
            if span.predicted_entity is not None:
                output = f"""<s><triplet> {span.text} <subj>"""
                model_outputs = tokenizer(output, max_length=1024, padding=True, truncation=True, return_tensors = 'pt', add_special_tokens=False)
                generated_tokens = model.generate(
                    model_inputs["input_ids"].to(model.device),
                    decoder_input_ids=model_outputs["input_ids"].to(model.device),
                    attention_mask=model_inputs["attention_mask"].to(model.device),
                    bad_words_ids=tokenizer(["<triplet>"], add_special_tokens=False).input_ids, # don't generate <triplet>
                    **gen_kwargs,
                )

                decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)

                for _,sentence in enumerate(decoded_preds):
                    new_triplets += KGBuilder.extract_triplets(sentence)

        
        return KGBuilder.filter_relationships(new_triplets)


    def build_graph(self,text:str):
        triplets = KGBuilder.relation_extraction(text)
        spans_el_base = KGBuilder.entity_linking(text)
        triplets_base = triplets.copy()
        spans_el_base_copy = spans_el_base.copy()

        doc,tns = KGBuilder.preprocess_doc_el(text)
        spans_from_re = KGBuilder.extract_entities_from_relation_extraction(triplets,doc.doc_id,text)
        spans_re = KGBuilder.process_doc_el(doc,tns,spans_from_re)
        spans_re_copy = spans_re.copy()
        spans_re = KGBuilder.filter_spans(spans_re)
        

        spans = KGBuilder.filter_spans(spans_el_base + spans_re)
        triplets,spans_no_relationship = KGBuilder.link_spans_with_relationships(triplets,spans)

        new_triplets = KGBuilder.extract_triplets_from_spans_with_no_relationship(text,spans_no_relationship)
        triplets_base_er = new_triplets.copy()
        new_triplets = KGBuilder.link_spans_with_relationships(new_triplets,spans)[0]
        triplets = triplets + new_triplets
        
        return [spans,triplets,spans_el_base_copy,spans_re_copy,triplets_base,triplets_base_er]
    
    def generate_embedding(token_id_values,attention_mask_values,token_type_values):
        output = KGBuilder.refined.model.transformer(
                        input_ids=token_id_values,
                        attention_mask=attention_mask_values,
                        token_type_ids=token_type_values,
                        position_ids=None,
                        head_mask=None,
                        inputs_embeds=None,
            )

        contextualised_embeddings = output.last_hidden_state
        
        return contextualised_embeddings
    
    def filter_mask_and_sums_from_spans(batch_elements,spans,device):
        ## TO-DO
        for batch_elem in batch_elements:
            index_start_batch = batch_elem.tokens[0].start
            index_end_batch = batch_elem.tokens[len(batch_elem.tokens)-1].end
            spans_in_batch = []

            for span in spans:
                if span.start>= index_start_batch and span.start + len(span.text)-1 <= index_end_batch:
                    spans_in_batch.append(span)

            batch_elem.add_spans(spans_in_batch)

        acc_sums_lst = [
                    [0] + list(map(lambda token: token.acc_sum, elem.tokens)) + [0]
                    for elem in batch_elements
                ]

        max_seq = max([len(batch_elem.tokens) + 2 for batch_elem in batch_elements])
        acc_sums = torch.tensor(
            pad(acc_sums_lst, seq_len=max_seq, pad_value=0), device=device, dtype=torch.long
        )

        b_entity_mask_lst = [elem.entity_mask for elem in batch_elements]
        b_entity_mask = torch.tensor(
            pad(b_entity_mask_lst, seq_len=-1, pad_value=0), device=device, dtype=torch.long
        )

        return acc_sums,b_entity_mask
    
    def filter_candidates(batch_elements,device):
        pem_values: List[List[float]] = []
        candidate_qcodes: List[str] = []
        candidate_qcodes_ints: List[List[int]] = []
        for batch_elem in batch_elements:
            for span in batch_elem.spans:
                pem_values.append(
                    [pem_value for _, pem_value in span.candidate_entities]
                )  # TODO unpad and pad here
                candidate_qcodes.extend(
                    [qcode for qcode, _ in span.candidate_entities]
                )  # should pad here
                # temporary hack (use negative IDs for additional entities IDs to avoid
                # collisions with Wikdata IDs
                candidate_qcodes_ints.append(
                    [int(qcode.replace("Q", "")) if 'Q' in qcode else int(qcode.replace("A", '-')) for qcode, _ in
                        span.candidate_entities]
                )

        num_cands = KGBuilder.refined.preprocessor.max_candidates
        num_ents = len([span for batch_elm in batch_elements for span in batch_elm.spans])
        cand_class_idx = KGBuilder.refined.preprocessor.get_classes_idx_for_qcode_batch(
                    candidate_qcodes, shape=(num_ents, num_cands, -1)
        )

        b_cand_desc_emb = None
        b_cand_desc = None

        b_cand_desc_emb = KGBuilder.refined.preprocessor.get_descriptions_emb_for_qcode_batch(
                        candidate_qcodes, shape=(num_ents, num_cands, -1)
                    ).to(device)
        b_cand_desc = None


        b_candidate_classes = torch.zeros(
                    size=(num_ents, num_cands, KGBuilder.refined.preprocessor.num_classes+1), dtype=torch.float32, device=device
        )
        first_idx = (
            torch.arange(num_ents, device=device)
                .unsqueeze(1)
                .unsqueeze(1)
                .expand(cand_class_idx.size())
        )
        snd_idx = torch.arange(num_cands, device=device).unsqueeze(1)
        b_candidate_classes[first_idx, snd_idx, cand_class_idx] = 1
        b_pem_values = torch.tensor(pem_values, device=device, dtype=torch.float32)
        b_candidate_qcode_values = torch.tensor(
            candidate_qcodes_ints, device=device, dtype=torch.long
        )

        return (b_candidate_qcode_values,b_pem_values,b_candidate_classes,b_cand_desc,b_cand_desc_emb)
    
    def call_model(batch,contextualised_embeddings,acc_sums,b_entity_mask,cand_desc,cand_desc_emb,candidate_pem_values,candidate_classes,device,spans,cand_ids)-> ModelReturn:
        mention_embeddings = KGBuilder.refined.model._get_mention_embeddings(
            sequence_output=contextualised_embeddings,
            token_acc_sums=acc_sums,
            entity_mask=b_entity_mask,
        )

        candidate_entity_targets = batch.candidate_target_values


        class_targets = KGBuilder.refined.model._expand_class_targets(
                    batch.class_target_values, index_tensor=batch.entity_index_mask_values
        )

        description_loss, candidate_description_scores = KGBuilder.refined.model.ed_2(
        candidate_desc=cand_desc,
        mention_embeddings=mention_embeddings,
        candidate_entity_targets=candidate_entity_targets,
        candidate_desc_emb=cand_desc_emb,
        )

        # forward pass of entity typing layer (using predetermined spans if provided else span identified by md layer)
        et_loss, et_activations = KGBuilder.refined.model.entity_typing(
        mention_embeddings=mention_embeddings, span_classes=class_targets
        )

        # forward pass of entity disambiguation layer
        ed_loss, ed_activations = KGBuilder.refined.model.entity_disambiguation(
        class_activations=et_activations.detach() if KGBuilder.refined.model.detach_ed_layer else et_activations,
        candidate_entity_targets=candidate_entity_targets,
        candidate_pem_values=candidate_pem_values,
        candidate_classes=candidate_classes,
        candidate_description_scores=candidate_description_scores.detach(),  # detach or not
        current_device=device,
        )
        
        return ModelReturn(
            None,
            None,
            et_loss,
            et_activations,
            ed_loss,
            ed_activations,
            spans,
            None,
            cand_ids,
            description_loss,
            candidate_description_scores,
        )

    def process_doc_el(doc:Doc,tns:Iterable[BatchedElementsTns],spans:List[Span]):
        for batch_idx,batch in enumerate(tns):
            ## MAYBE THIS CAUSE ERROR
            batch_elements = batch.batch_elements

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            token_id_values = batch.token_id_values.to(device)
            attention_mask_values = batch.attention_mask_values.to(device)
            token_type_values = batch.token_type_values.to(device)

            contextualised_embeddings = KGBuilder.generate_embedding(token_id_values,attention_mask_values,token_type_values)

            person_coreference = dict()

            person_coreference = KGBuilder.refined.preprocessor.add_candidates_to_spans(
                spans,person_coreference=person_coreference
            )
            acc_sums,b_entity_mask = KGBuilder.filter_mask_and_sums_from_spans(batch_elements,spans,device)

            (cand_ids,
            candidate_pem_values,
            candidate_classes,
            cand_desc,
            cand_desc_emb ) = KGBuilder.filter_candidates(batch_elements,device)

            output = KGBuilder.call_model(batch,contextualised_embeddings,acc_sums,b_entity_mask,cand_desc,cand_desc_emb,candidate_pem_values,candidate_classes,device,spans,cand_ids)


            spans = output.entity_spans

            cand_ids = torch.cat(
                        [output.cand_ids, torch.ones((output.cand_ids.size(0), 1), device=device, dtype=torch.long) * -1], 1
                    )

            ed_targets_predictions = output.ed_activations.argmax(dim=1)
            ed_targets_softmax = output.ed_activations.softmax(dim=1)


            description_scores = output.candidate_description_scores.detach().cpu().numpy()

            predicted_entity_ids = (
                cand_ids[torch.arange(cand_ids.size(0)), ed_targets_predictions].cpu().numpy().tolist()
            )
            predicted_entity_confidence = round_list(
                ed_targets_softmax[torch.arange(ed_targets_softmax.size(0)), ed_targets_predictions]
                    .cpu().detach()
                    .numpy()
                    .tolist(),
                4,
            )


            span_to_classes = defaultdict(list)
            span_indices, pred_class_indices = torch.nonzero(
                output.et_activations > 0.5, as_tuple=True
            )
            for span_idx, pred_class_idx, conf in zip(
                    span_indices.cpu().numpy().tolist(),
                    pred_class_indices.cpu().numpy().tolist(),
                    round_list(
                        output.et_activations[(span_indices, pred_class_indices)].cpu().detach().numpy().tolist(), 4
                    ),
            ):
                if pred_class_idx == 0:
                    continue  # skip padding class label
                class_id = KGBuilder.refined.preprocessor.index_to_class.get(pred_class_idx, "Q0")
                class_label = KGBuilder.refined.preprocessor.class_to_label.get(class_id, "no_label")
                span_to_classes[span_idx].append((class_id, class_label, conf))

            sorted_entity_ids_scores, old_indices = ed_targets_softmax.sort(descending=True)
            sorted_entity_ids_scores = sorted_entity_ids_scores.cpu().detach().numpy().tolist()
            sorted_entity_ids = KGBuilder.refined.sort_tensor(cand_ids, old_indices).cpu().numpy().tolist()


            for span_idx, span in enumerate(spans):
                wikidata_id = f'Q{str(predicted_entity_ids[span_idx])}'
                span.predicted_entity = Entity(
                    wikidata_entity_id=wikidata_id,
                    wikipedia_entity_title=KGBuilder.refined.preprocessor.qcode_to_wiki.get(wikidata_id)
                    if KGBuilder.refined.preprocessor.qcode_to_wiki is not None else None
                )
                span.entity_linking_model_confidence_score = predicted_entity_confidence[span_idx]
                span.top_k_predicted_entities = [
                    (Entity(wikidata_entity_id=f'Q{entity_id}',
                            wikipedia_entity_title=KGBuilder.refined.preprocessor.qcode_to_wiki.get(wikidata_id)
                            if KGBuilder.refined.preprocessor.qcode_to_wiki is not None else None
                            ),
                        round(score, 4))
                    for entity_id, score in
                    zip(sorted_entity_ids[span_idx], sorted_entity_ids_scores[span_idx])
                    if entity_id != 0
                ]

                span.candidate_entities = [
                    (qcode, round(conf, 4))
                    for qcode, conf in filter(lambda x: not x[0] == "Q0", span.candidate_entities)
                ]
                span.description_scores = round_list(
                    description_scores[span_idx].tolist(), 4
                )  # matches candidate order
                span.predicted_entity_types = span_to_classes[span_idx]
        
        
        return spans

    def preprocess_doc_el(text:str):
        doc = Doc.from_text(text,
                    preprocessor=KGBuilder.refined.preprocessor)
        tns: Iterable[BatchedElementsTns] = convert_doc_to_tensors(
                    doc,
                    KGBuilder.refined.preprocessor,
                    collate=True,
                    max_batch_size=16,
                    sort_by_tokens=False,
                    max_seq=KGBuilder.refined.max_seq,
            )
        
        return (doc,tns) 
        
    def extract_entities_from_relation_extraction(triplets,doc_id,text:str)->List[Span]:
        entity_mentions = set()
        for triplet in triplets:
            entity_mentions.add(triplet['head'])
            entity_mentions.add(triplet['tail'])
        
        spans: List[Span] = []

        for ent in entity_mentions:
            indexes_start = KGBuilder.__find_all(ent,text)
            for index_start in indexes_start:
                span = Span(
                    start = index_start,
                    ln = len(ent),
                    text = ent,
                    coarse_type=None,
                    coarse_mention_type=None,
                    doc_id=doc_id
                )

                spans.append(span)
        
        
        return spans

    def extract_triplets(text:str)->List[Dict]:
        triplets = []
        relation, subject, relation, object_ = '', '', '', ''
        text = text.strip()
        current = 'x'
        for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
            if token == "<triplet>":
                current = 't'
                if relation != '':
                    triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
                    relation = ''
                subject = ''
            elif token == "<subj>":
                current = 's'
                if relation != '':
                    triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
                object_ = ''
            elif token == "<obj>":
                current = 'o'
                relation = ''
            else:
                if current == 't':
                    subject += ' ' + token
                elif current == 's':
                    object_ += ' ' + token
                elif current == 'o':
                    relation += ' ' + token
        if subject != '' and relation != '' and object_ != '':
            triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
        return triplets
