# # Example usage
import subprocess
import pandas as pd
from transformers import pipeline


import pytesseract
from bluemist.environment import initialize

# Print the version of pytesseract
print("Pytesseract version:", pytesseract.get_tesseract_version())

from bluemist.llm import api_wrapper
from bluemist.llm.wrapper import perform_task

initialize()

# task = "question-answering"
# input = "My name is Sarah and I live in London"
# question = "Where does Sarah live?"
# #df = perform_task(task, input, question, override_models=['deepset/roberta-base-squad2-distilled', 'deepset/roberta-base-squad2'], limit=2)
# df = perform_task(task, input, question, limit=1)

task = "document-question-answering"
input_data = "https://templates.invoicehome.com/invoice-template-us-neat-750px.png"
question = "What is the ship to address?"
limit = 1
evaluate_models=True
limit = 1
df = perform_task(task, input_data, question, limit=limit, evaluate_models=evaluate_models)
print(df)

# task = "summarization"
# task = "sentiment-analysis"
# input = "During the period of Delhi Sultanate, which covered most of today's north India, eastern Pakistan, southern Nepal and Bangladesh[46] and which resulted in the contact of Hindu and Muslim cultures, the Sanskrit and Prakrit base of Old Hindi became enriched with loanwords from Persian, evolving into the present form of Hindustani.[47][48][49][50][51][52] The Hindustani vernacular became an expression of Indian national unity during the Indian Independence movement,[53][54] and continues to be spoken as the common language of the people of the northern Indian subcontinent,[55] which is reflected in the Hindustani vocabulary of Bollywood films and songs.[56][57]"
# df = perform_task(task, input, limit=1)
# print(df)

#summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
#print(summarizer(ARTICLE, max_length=130, min_length=30, do_sample=False))

# df = perform_task(task, ARTICLE, evaluate_models=True, override_models='facebook/bart-large-cnn', limit=1)
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#     print(df)

# ARTICLE = ["there is a shortage of capital, and we need extra financing",
#              "growth is strong and we have plenty of liquidity",
#              "there are doubts about our finances",
#              "profits are flat"]

# df = perform_task(task, ARTICLE, evaluate_models=True, limit=2)
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#     print(df)


from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

# finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
# tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
#
# nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)
#
# sentences = ["there is a shortage of capital, and we need extra financing",
#              "growth is strong and we have plenty of liquidity",
#              "there are doubts about our finances",
#              "profits are flat"]
# results = nlp(sentences)
# print(results)  #LABEL_0: neutral; LABEL_1: positive; LABEL_2: negative

#api_wrapper.start_api_server(port=8001)
