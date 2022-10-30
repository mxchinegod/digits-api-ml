from fastapi import FastAPI, Body
from typing import Dict, Any

app = FastAPI()

@app.post("/gpt2")
async def gpt2(query: str = Body(embed=True)):
  """
  This is using the HuggingFace transformers library to load a pretrained model and
  tokenizer. The model is then used to answer the question.
  """
  from transformers import pipeline, set_seed
  generator = pipeline('text-generation', model='gpt2')
  set_seed(42)
  output = generator(query, max_length=30, num_return_sequences=5)
  return {"message": output}

@app.post("/roberta")
async def roberta(query: Dict[Any, Any]):
  """
  The above code is using the HuggingFace transformers library to load a pretrained model and
  tokenizer. The model is then used to answer the question.
  """
  from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
  model_name = "deepset/roberta-base-squad2"
  nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
  QA_input = query
  output = nlp(QA_input)
  AutoModelForQuestionAnswering.from_pretrained(model_name)
  AutoTokenizer.from_pretrained(model_name)
  return {"message": output}

@app.post("/bart_cnn")
async def bart_cnn(query: Dict[Any, Any]):
  """
  > This function takes a query, which is a dictionary with a key called "query" and a value that is a
  string. It then uses the transformers library to summarize the query and returns a dictionary with a
  key called "message" and a value that is the summary
  
  :param query: The text to be summarized
  :type query: Dict[Any, Any]
  :return: A dictionary with a key "message" and a value of the output of the summarizer.
  """
  import json
  query = json.dumps(query["query"])
  print(query)
  from transformers import pipeline
  summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
  output = summarizer(query[0:1023], max_length=len(query)/2, min_length=30, do_sample=False)
  return {"message": output}