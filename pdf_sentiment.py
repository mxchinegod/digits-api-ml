import json
import requests
import io
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
import re
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification


def convert_pdf_to_txt(url):
    """
    It takes a URL, downloads the PDF, converts it to text, and returns the text

    :param url: The URL of the PDF file
    :return: A string of text
    """
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle)
    page_interpreter = PDFPageInterpreter(resource_manager, converter)
    with requests.get(url) as response:
        content = io.BytesIO(response.content)
        for page in PDFPage.get_pages(content, caching=True, check_extractable=True):
            page_interpreter.process_page(page)
    text = fake_file_handle.getvalue()
    return text


def classify_sentiment(sentence, model, tokenizer):
    """
    It takes a sentence, tokenizes it, and then passes it through the model

    :param sentence: the sentence to classify
    :param model: the model we're using to classify the sentiment
    :param tokenizer: The tokenizer that we used to train the model
    :return: The last hidden state of the model.
        `main` takes a URL as an argument, converts the PDF to text, splits the text into sentences, and
        then classifies each sentence as positive, negative, or neutral

        :param url: the url of the pdf you want to analyze
        :return: A list of sentiments
    """
    inputs = tokenizer.encode_plus(
        sentence, return_tensors="pt", add_special_tokens=True)
    input_ids = inputs["input_ids"].to(torch.device("cpu"))
    outputs = model(input_ids)
    last_hidden_states = outputs[0]
    return last_hidden_states.detach().numpy()[0]

def main(url):
    text = convert_pdf_to_txt(url)
    sentences = re.split(r'[.!?]', text)
    sentiments = []
    tokenizer = AutoTokenizer.from_pretrained(
        "nickmuchi/deberta-v3-base-finetuned-finance-text-classification")
    model = AutoModelForSequenceClassification.from_pretrained(
        "nickmuchi/deberta-v3-base-finetuned-finance-text-classification")
    for sentence in sentences:
        if len(sentence) > 50:
            sentiment = classify_sentiment(sentence, model, tokenizer)
            if sentiment[0] > sentiment[1] and sentiment[0] > sentiment[2]:
                flat = '🔥'
                color = 'red'
            elif sentiment[1] > sentiment[0] and sentiment[1] > sentiment[2]:
                flat = '👌🏼'
                color = 'grey'
            elif sentiment[2] > sentiment[1] and sentiment[2] > sentiment[0]:
                flat = '🚀'
                color = 'green'
            sentiment_dict = {"sentence": sentence, "sentiment": sentiment.tolist(), "flat": flat, "color": color}
            sentiments.append(sentiment_dict)
    return json.dumps(sentiments)