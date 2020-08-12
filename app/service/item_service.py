from models.item_model import Payload
from models.item_model import Question

import pandas as pd
import requests
import uuid
import torch

cuda_available = torch.cuda.is_available()

def index_item(payload: Payload):
    print(payload)
    return payload

def ask_question(question: Question):
    print(question)
    return question