import requests
import io
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
import re
import torch
import termcolor
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
        print(termcolor.colored(sentence, color), flat)
    return sentiments


# It's running the main function with the url as an argument.
if __name__ == "__main__":
    urls = [
        "https://www.federalreserve.gov/monetarypolicy/files/monetary20230201a1.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/monetary20220126a1.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/fomcminutes20220126.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/monetary20220316a1.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/fomcprojtabl20220316.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/fomcminutes20220316.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/monetary20220504a1.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/fomcminutes20220504.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/monetary20220615a1.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/fomcprojtabl20220615.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/fomcminutes20220615.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/monetary20220727a1.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/fomcminutes20220727.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/monetary20220921a1.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/fomcprojtabl20220921.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/fomcminutes20220921.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/monetary20221102a1.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/fomcminutes20221102.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/monetary20221214a1.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/fomcprojtabl20221214.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/fomcminutes20221214.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/monetary20210127a1.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/fomcminutes20210127.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/monetary20210317a1.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/fomcprojtabl20210317.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/fomcminutes20210317.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/monetary20210428a1.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/fomcminutes20210428.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/monetary20210616a1.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/fomcprojtabl20210616.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/fomcminutes20210616.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/monetary20210728a1.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/fomcminutes20210728.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/monetary20210922a1.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/fomcprojtabl20210922.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/fomcminutes20210922.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/monetary20211103a1.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/fomcminutes20211103.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/monetary20211215a1.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/fomcprojtabl20211215.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/fomcminutes20211215.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/monetary20200129a1.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/fomcminutes20200129.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/monetary20200303a1.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/monetary20200315a1.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/fomcminutes20200315.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/monetary20200323a1.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/monetary20200429a1.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/fomcminutes20200429.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/monetary20200610a1.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/fomcprojtabl20200610.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/fomcminutes20200610.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/monetary20200729a1.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/fomcminutes20200729.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/monetary20200916a1.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/fomcprojtabl20200916.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/fomcminutes20200916.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/monetary20201105a1.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/fomcminutes20201105.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/monetary20201216a1.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/fomcprojtabl20201216.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/fomcminutes20201216.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/monetary20190130a1.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/fomcminutes20190130.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/monetary20190320a1.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/fomcprojtabl20190320.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/fomcminutes20190320.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/monetary20190501a1.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/fomcminutes20190501.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/monetary20190619a1.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/fomcprojtabl20190619.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/fomcminutes20190619.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/monetary20190731a1.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/fomcminutes20190731.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/monetary20190918a1.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/fomcprojtabl20190918.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/fomcminutes20190918.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/monetary20191011a1.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/monetary20191030a1.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/fomcminutes20191030.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/monetary20191211a1.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/fomcprojtabl20191211.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/fomcminutes20191211.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/monetary20180131a1.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/fomcminutes20180131.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/monetary20180321a1.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/fomcprojtabl20180321.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/fomcminutes20180321.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/monetary20180502a1.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/fomcminutes20180502.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/monetary20180613a1.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/fomcprojtabl20180613.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/fomcminutes20180613.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/monetary20180801a1.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/fomcminutes20180801.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/monetary20180926a1.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/fomcprojtabl20180926.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/fomcminutes20180926.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/monetary20181108a1.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/fomcminutes20181108.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/monetary20181219a1.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/fomcprojtabl20181219.pdf",
        "https://www.federalreserve.gov/monetarypolicy/files/fomcminutes20181219.pdf"
    ]
    for url in urls:
        sentiments = main(url)
