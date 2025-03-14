import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, TextStreamer
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline

# ========== CUDA MEMORY CHECK ==========
#print(torch.cuda.memory_summary())  # optional

# ========== MODEL SETUP ==========
model_name_or_path = "TheBloke/Llama-2-7B-GPTQ"
DEVICE = "cuda"

# Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    device_map="auto",  # Automatically place layers across GPUs
    torch_dtype=torch.float16,  # GPTQ works best in fp16
    trust_remote_code=True
)

# Use streamer for streaming output (optional)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# Create HF Pipeline for LangChain
text_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.0,
    do_sample=False,
    top_p=1.0,
    repetition_penalty=1.15,
    streamer=streamer,  # optional
)

llm = HuggingFacePipeline(pipeline=text_pipeline)

# ========== EMBEDDING SETUP ==========
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large",
    model_kwargs={"device": DEVICE}
)

# ========== VECTORSTORE SETUP ==========
db = Chroma(persist_directory="db", embedding_function=embeddings)

# ========== PROMPT TEMPLATE ==========
template = """
Answer the question based only on the provided context.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}

Answer:"""

prompt = PromptTemplate(template=template, input_variables=["context", "question"])

# ========== QA CHAIN ==========
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 1}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt},
)
