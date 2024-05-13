import os
import pinecone
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from langchain.vectorstores import Pinecone
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAILogicModel
from langchain.prompts import PromptTemplate

# Initialize Pinecone client
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"))

# Create a Pinecone index
index_name = "wordpress-chatbot-index"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name=index_name, metric="cosine")

# Set up the vector database
vector_db = Pinecone(
    index_name=index_name,
    embedding=HuggingFaceEmbeddings(),
)

# Set up the RAG-based chatbot
class RAGChatbot:
    def __init__(self):
        self.vector_db = vector_db
        self.qa_chain = None

    def setup(self, model_id: str, tokenizer_id: str):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        self.model = AutoModelForMaskedLM.from_pretrained(model_id)

    def generate_embedding(self, text: str):
        tokens = self.tokenizer(text, return_tensors="pt")
        output = self.model(**tokens)
        vec = torch.max(torch.log(1 + torch.relu(output.logits)) * tokens.attention_mask.unsqueeze(-1), dim=1)[0].squeeze()
        cols = vec.nonzero().squeeze().cpu().tolist()
        weights = vec[cols].cpu().tolist()
        sparse_dict = dict(zip(cols, weights))
        return sparse_dict

    def add_data_to_vector_db(self, data: list):
        texts = [item["text"] for item in data]
        embeddings = [self.generate_embedding(text) for text in texts]
        self.vector_db.insert(ids=[str(item["id"]) for item in data], vectors=embeddings)

    def set_up_qa_chain(self, chain_type: str):
        if self.qa_chain is not None:
            return

        # Set up the LLM
        llm = OpenAILogicModel(temperature=0, max_tokens=256)

        # Set up the prompt
        if chain_type == "cot":
            prompt = PromptTemplate(
                input_variables=["page_content", "question"],
                template="{page_content}\n\nQuestion: {question}\nAnswer:"
            )
        else:
            prompt = PromptTemplate(
                input_variables=["context", "question"],
                template="Context: {context}\n\nQuestion: {question}\nAnswer:"
            )

        # Set up the QA chain
        self.qa_chain = load_qa_chain(llm, prompt, self.vector_db.as_retriever())

    def ask_question(self, question: str, chain_type: str):
        if self.qa_chain is None:
            self.set_up_qa_chain(chain_type)

        return self.qa_chain({"question": question})

# Example usage
chatbot = RAGChatbot()

# Add data to the vector database
data = [
    {"id": 1, "text": "This is a sample text for the chatbot."},
    {"id": 2, "text": "Another example text for the chatbot to learn from."}
]

chatbot.add_data_to_