from fastapi import FastAPI, Body
import json
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
    The below code is using the HuggingFace transformers library to load a pretrained model and
    tokenizer. The model is then used to answer the question.
    """
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
    model_name = "deepset/roberta-base-squad2"
    nlp = pipeline('question-answering',
                   model=model_name, tokenizer=model_name)
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
    query = json.dumps(query["query"])
    from transformers import pipeline
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    output = summarizer(query[0:2023], max_length=len(
        query)/2, min_length=30, do_sample=False)
    return {"message": output}


@app.post('/autodd')
async def autodd():
    from AutoDD import acquire
    data = acquire(True)
    if len(data) > 0:
        return data


@app.post('/greeks')
async def greeks(query: Dict[Any, Any]):
    query = query["query"]
    from options import option
    # otype = Call or Put
    # S0 = Underlying Price
    # K = Strike
    # T = days to expiration or date (YYYY-MM-DD)
    # ls = Long or Short
    # vol = Volatility
    # marketPrice = Options Market Price
    # q = Dividend Yield
    # r = Risk Free Rate
    # data["gamma"] = option(otype='Put', S0=227.44, K=200, expDay='2022-11-18', ls="Long", vol=29.00, marketPrice=0.34, q=1.19, r=0.045).gamma(S0=227.44, K=200, vol=29.00, q=1.19, r=0.045)
    _ = option(otype=query['otype'], S0=query['S0'], K=query['K'], expDay=query['expDay'],
               ls=query['ls'], marketPrice=query['marketPrice'], q=query['q'], r=query['r'])
    sweep = _.sweep({"price": [query['marketPrice']-50,
                    query['marketPrice']+50, 50], "vol": [0.1, 0.5, 50]}, "ultima")
    return {
        "price": sweep['price'].tolist(), "vol": sweep['vol'].tolist(), 'S0': sweep['S0'], 'K': sweep['K'], 'r': sweep['r'], 'T': sweep['T'].tolist(), 'q': sweep['q'], 'ultima': sweep['ultima'].tolist(), 'ls': sweep['ls']
    }


@app.post('/oracle')
async def agi(query: Dict[Any, Any]):
    query = query["query"]
    import AGI
    return { "answer": AGI.main(query) }
