import inspect
from packaging import version
from ratransformers import RATransformer
from transformers import AutoModelForSeq2SeqLM
import transformers


class TestModelsForward:
    """
    Test small supported models to see if nothing is breaking in forward step
    """
    def ratransformer_forward(self, ratransformer: RATransformer):
        model = ratransformer.model
        tokenizer = ratransformer.tokenizer
        encoding = tokenizer(
            "this is just a dummy text",
            return_tensors="pt",
            input_relations=None
        )
        if 'decoder_input_ids' in inspect.signature(model.forward).parameters:
            if version.parse(transformers.__version__) >= version.parse('4.13'):
                decoder_input_ids = model._prepare_decoder_input_ids_for_generation(encoding['input_ids'].shape[0], None, None)
            else:
                decoder_input_ids = model._prepare_decoder_input_ids_for_generation(encoding['input_ids'], None, None)
            encoding['decoder_input_ids'] = decoder_input_ids
        _ = model(**encoding)

    def test_t5(self):
        ratransformer = RATransformer(
            "t5-small",
            relation_kinds=['dummy1', 'dummy2'],
            model_cls=AutoModelForSeq2SeqLM
        )
        self.ratransformer_forward(ratransformer)