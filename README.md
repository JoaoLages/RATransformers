<div align="center">

# RATransformers 🐭

![PyPI - Latest Package Version](https://img.shields.io/pypi/v/ratransformers?logo=pypi&style=flat&color=orange) ![GitHub - License](https://img.shields.io/github/license/JoaoLages/ratransformers?logo=github&style=flat&color=green)

**RATransformers**, short for Relation-Aware Transformers, is a package built on top of [transformers 🤗](https://github.com/huggingface/transformers)
that enables the training/fine-tuning of models with extra relation-aware input features.
</div>

### Example - Encoding a table in TableQA (Question Answering on Tabular Data)
![](assets/tableQA.gif)

[[Notebook Link](https://github.com/JoaoLages/RATransformers/blob/main/notebooks/TableQA_tabfact_example.ipynb)]

In this example we can see that passing the table as text with no additional information to the model is a poor representation.

With RATransformers 🐭 you are able to encode the table in a more structured way by passing specific relations within the input.
RATransformers 🐭 also allows you to pass further features related with each input word/token.

Check more examples in [[here](https://github.com/JoaoLages/RATransformers/blob/main/notebooks/)].

## Installation

Install directly from PyPI:

    pip install ratransformers

## Usage

```python
from ratransformers import RATransformer
from transformers import AutoModelForSequenceClassification


ratransformer = RATransformer(
    "nielsr/tapex-large-finetuned-tabfact", # define the 🤗 model you want to load
    relation_kinds=['is_value_of_column', 'is_from_same_row'], # define the relations that you want to model in the input
    model_cls=AutoModelForSequenceClassification, # define the model class
    pretrained_tokenizer_name_or_path='facebook/bart-large' # define the tokenizer you want to load (in case it is not the same as the model)
)
model = ratransformer.model
tokenizer = ratransformer.tokenizer
```

With only these steps your RATransformer 🐭 is ready to be trained. 

More implementation details in [the examples here](https://github.com/JoaoLages/RATransformers/blob/main/notebooks/).

## How does it work?
We modify the self-attention layers of the transformer model as explained in the section 3 of [the RAT-SQL paper](https://arxiv.org/pdf/1911.04942.pdf).

## Supported Models
Currently we support a limited number of transformer models:
- [BART](https://huggingface.co/docs/transformers/model_doc/bart)
- [BERT](https://huggingface.co/docs/transformers/model_doc/bert)
- [GPT-2](https://huggingface.co/docs/transformers/model_doc/gpt2)
- [RoBERTa](https://huggingface.co/docs/transformers/model_doc/roberta)
- [T5](https://huggingface.co/docs/transformers/model_doc/t5)
- [LongT5](https://huggingface.co/docs/transformers/model_doc/longt5)

Want another model? Feel free to open an [Issue](https://github.com/JoaoLages/RATransformers/issues) or create a [Pull Request](https://github.com/JoaoLages/RATransformers/pulls) and let's get started 🚀
