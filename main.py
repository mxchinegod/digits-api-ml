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