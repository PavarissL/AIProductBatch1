import gradio as gr
from rag_pipeline import qa_chain  # โหลด chain ที่เตรียมไว้
import re

# 🔧 ฟังก์ชันสำหรับตัด prompt ที่ model ตอบกลับมาทั้งหมด
def clean_model_response(response: str) -> str:
    match = re.search(r"Answer:\s*(.*)", response, re.DOTALL)
    return match.group(1).strip() if match else response.strip()

# 🎯 ฟังก์ชันหลักที่ตอบคำถาม
def answer_question(query):
    try:
        result = qa_chain.invoke(query)
        raw_answer = result.get("result", "")
        answer = clean_model_response(raw_answer)

        # ดึงแหล่งอ้างอิง (source) ถ้ามี
        sources = "\n".join([doc.metadata.get("source", "Unknown") for doc in result.get("source_documents", [])])
        return answer, sources
    except Exception as e:
        return f"❌ Error: {str(e)}", ""

# 🌐 Gradio UI
with gr.Blocks(title="📄 LLaMA 2 PDF Q&A") as demo:
    gr.Markdown("## 🤖 ถาม-ตอบข้อมูลจากเอกสาร PDF ด้วย LLaMA 2 + LangChain")

    with gr.Row():
        query_input = gr.Textbox(
            label="❓ คำถามของคุณ",
            placeholder="พิมพ์คำถาม เช่น 'รายได้รวมของบริษัทในปี 2023 เท่าไหร่?'",
            lines=2,
            show_label=True,
        )
        submit_btn = gr.Button("📨 ถามเลย", scale=0)

    with gr.Row():
        answer_output = gr.Textbox(label="✍️ คำตอบ", lines=5, interactive=False)
        source_output = gr.Textbox(label="📚 แหล่งข้อมูลอ้างอิง", lines=3, interactive=False)

    submit_btn.click(fn=answer_question, inputs=[query_input], outputs=[answer_output, source_output])

# ถ้าใช้ Docker หรือ VM: server_name="0.0.0.0" จะช่วยให้เข้าผ่าน IP ได้
# demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
demo.launch()
