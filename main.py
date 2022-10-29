from fastapi import FastAPI, Body
from typing import Dict, Any

app = FastAPI()

@app.post("/gpt2")
async def gpt2(query: str = Body(embed=True)):
  from transformers import pipeline, set_seed
  generator = pipeline('text-generation', model='gpt2')
  set_seed(42)
  output = generator(query, max_length=30, num_return_sequences=5)
  return {"message": output}

@app.post("/roberta")
async def roberta(query: Dict[Any, Any]):
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
  query = query["query"][0:4096]
  from transformers import pipeline
  summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
  output = summarizer(query, max_length=len(query)/2, min_length=30, do_sample=False)
  return {"message": output}