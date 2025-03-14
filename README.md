# 🦙 LLaMA-2 7B GPTQ RAG (Retrieval-Augmented Generation) Pipeline

## 🇬🇧 English Description

This project implements a RAG (Retrieval-Augmented Generation) system using:

- 🤖 **LLaMA-2 7B GPTQ** from TheBloke (quantized for fast inference)
- 🔗 **LangChain** for orchestration
- 🧠 **HuggingFace Transformers** for model and tokenizer
- 🧠 **Multilingual-E5 Embeddings** for document representation
- 📦 **Chroma DB** for vector storage
- 💡 **RetrievalQA chain** for answering questions based on context

### 🔍 What it does

This pipeline allows users to ask questions, and the system will:
1. Retrieve the most relevant context from a document database.
2. Generate an answer using LLaMA-2 based only on that context.

If the answer isn't found in the context, the model will say:
> "I don't know."

### ✅ Features
- GPU-accelerated with CUDA
- GPTQ support for memory-efficient inference
- Streamed generation output (optional)
- Easy to integrate into apps using LangChain

---

## 🇹🇭 รายละเอียดภาษาไทย

โปรเจกต์นี้เป็นระบบ RAG (Retrieval-Augmented Generation) ที่ใช้เทคโนโลยี:

- 🤖 **LLaMA-2 7B GPTQ** จาก TheBloke (เวอร์ชันที่ถูก quantize เพื่อให้รันได้เร็วและเบา)
- 🔗 **LangChain** สำหรับจัดการ pipeline
- 🧠 **HuggingFace Transformers** สำหรับโมเดลและ tokenizer
- 🧠 **Multilingual-E5 Embeddings** เพื่อใช้แปลงข้อความเป็นเวกเตอร์
- 📦 **Chroma DB** สำหรับเก็บเวกเตอร์
- 💡 **RetrievalQA chain** สำหรับตอบคำถามจาก context ที่ดึงมาได้

### 🔍 ระบบนี้ทำอะไร

เมื่อผู้ใช้ตั้งคำถาม ระบบจะ:
1. ดึงข้อมูลหรือ context ที่เกี่ยวข้องจากฐานข้อมูลเวกเตอร์
2. สร้างคำตอบจาก context นั้นโดยใช้ LLaMA-2

หากไม่สามารถหาคำตอบจาก context ได้ ระบบจะตอบว่า:
> "I don't know."

### ✅ คุณสมบัติเด่น
- รองรับ CUDA บน GPU
- ใช้ GPTQ ประหยัดแรมและเร็ว
- รองรับการแสดงผลแบบ stream (option)
- ใช้งานง่ายผ่าน LangChain API

---

## 🚀 Quick Start

```bash
# Clone the project
git clone <your-repo-url>
cd <your-project-folder>

# Create environment and install dependencies
conda create -n rag_llama2 python=3.10
conda activate rag_llama2
pip install -r requirements.txt

# (Optional) Install gptqmodel if using GPTQ
pip install --prefer-binary gptqmodel

# Run the app
python app_gradio.py

Folder Structure
├── app_gradio.py         # Main Gradio app
├── rag_pipeline.py       # LangChain pipeline setup
├── db/                   # Chroma vector database
├── requirements.txt      # All Python dependencies
└── README.md             # This file
