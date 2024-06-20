
from refined.inference.processor import Refined
from refined.evaluation.evaluation import eval_all

refined = Refined.from_pretrained(model_name='wikipedia_model_with_numbers',
                                  entity_set="wikipedia",use_precomputed_descriptions=True)

#spans = refined.process_text("England won the FIFA World Cup in 1966.")
spans = refined.process_text("Marie Curie was a pioneering physicist and chemist. She discovered radioactivity and won Nobel Prizes in both Physics and Chemistry.")
print(spans)
"""
from transformers import pipeline
from refined.inference.processor import Refined
import tqdm as notebook_tqdm


text = "Apple is a technology company. It founded by Steve Jobs and Steve Wozniak. The headquarters is in Cupertino California."

triplet_extractor = pipeline('text2text-generation',model='Babelscape/rebel-large',tokenizer='Babelscape/rebel-large')
output = triplet_extractor(text,return_tensors=True,return_text=False)

traced_text = triplet_extractor.tokenizer.batch_decode([output[0]["generated_token_ids"]])
#text_rebel_tokenized = triplet_extractor.tokenizer(text,return_tensors='pt')
#triplet_extractor.model.generate(text_rebel_tokenized['input_ids'])

refined = Refined.from_pretrained(model_name='wikipedia_model',
                                  entity_set='wikipedia',
                                  use_precomputed_descriptions=True)
print('EL results (with model fine-tuned on Wikipedia model)')
eval_all(refined=refined, el=True, filter_nil_spans=False)
"""