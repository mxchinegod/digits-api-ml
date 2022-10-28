from fastapi import FastAPI
from transformers import pipeline, set_seed

app = FastAPI()

@app.get("/gpt2")
async def gpt2():
  from transformers import pipeline, set_seed
  generator = pipeline('text-generation', model='gpt2')
  set_seed(42)
  output = generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5)

  return {"message": output}

@app.get("/roberta")
async def roberta():
  from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

  model_name = "deepset/roberta-base-squad2"

  # a) Get predictions
  nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
  QA_input = {
      'question': 'Will Elon make Twitter lose money?',
      'context': '''GM says it is "pausing" advertising as it evaluates Twitter's new direction. It will continue to utilize the platform to interact with customers but not pay for advertising. "We are engaging with Twitter to understand the direction of the platform under their new ownership," GM said.
"We have temporarily paused our paid advertising. Our customer care interactions on Twitter will continue," the company said in an emailed statement. Other auto companies, including Ford Motor, Stellantis and Alphabet-owned Waymo, did not immediately respond to requests for comment on whether they plan to suspend advertising.
Elon Musk says he will not reinstate any accounts or make major content decisions before a content moderation council convenes. Musk has said he is a "free speech absolutist" who would restore the account of former President Donald Trump. Electric truck maker Nikola said it had no plans to change anything regarding Twitter.
Fisker CEO Henrik Fisker deleted his Twitter account earlier this year when Twitter's board accepted Musk's bid to buy the company and take it private. "I don't want to be on Twitter anymore," he said at the time. "It's not a place for me. I don't like it."
Musk has long boasted that Tesla does not pay for traditional advertising. Instead, Tesla rewards people who run, or are members of, Tesla owners' clubs. Fisker Inc. continues to use Twitter, which every major automotive brand utilizes for customer engagement and marketing.'''
  }
  res = nlp(QA_input)

  # b) Load model & tokenizer
  model = AutoModelForQuestionAnswering.from_pretrained(model_name)
  tokenizer = AutoTokenizer.from_pretrained(model_name)

  return {"message": res}