import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import datetime
import os
import types
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# -----------------------------
# 🩹 Fix torch.classes error
if not isinstance(torch.classes, types.ModuleType):
    torch.classes = types.SimpleNamespace()

# -----------------------------
# 1️⃣ Streamlit Page Config
st.set_page_config(page_title="💬 Financial Chatbot ", layout="wide")
st.title("📊 Financial Chatbot ")

# -----------------------------

# -----------------------------
# 3️⃣ Device and Model Config
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "ibm-granite/granite-3.3-2b-base"
HF_TOKEN = os.getenv("hf_token")  # Or replace with your token directly

@st.cache_resource(show_spinner="Loading model...")
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_auth_token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        use_auth_token='give_user_auth_token',
    )
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# -----------------------------
# 4️⃣ Query function
def query_granite(user_question):
    try:
        prompt = f"You are a helpful financial assistant. Answer the following question:\n{user_question}"

        input_tokens = tokenizer(prompt, return_tensors="pt").to(device)
        output_tokens = model.generate(
            **input_tokens,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

        full_response = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]

        if prompt in full_response:
            full_response = full_response.replace(prompt, "").strip()

        for tag in ["USER:", "ASSISTANT:", "AGENT:", "Calling function with"]:
            full_response = full_response.replace(tag, "").strip()

        return full_response

    except Exception as e:
        return f"❌ Error generating response: {str(e)}"

# -----------------------------

st.markdown("---")

# -----------------------------
# 6️⃣ User Input + Chatbot
with st.container():
    st.subheader("💬 Ask the Chatbot")
    user_input = st.text_input("Ask a financial question:")
    send_button = st.button("Send")

    if send_button and user_input:
        response = query_granite(user_input)
        st.text_area("📥 Chatbot Response", response, height=300)

        # -----------------------------
        # 7️⃣ Export Chat Section
        with st.expander("📄 Export Chat to PDF"):
            filename = f"Chat_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            doc = SimpleDocTemplate(filename)
            styles = getSampleStyleSheet()
            elements = [
                Paragraph(f"<b>User:</b> {user_input}", styles['Normal']),
                Paragraph(f"<b>Bot:</b> {response}", styles['Normal'])
            ]
            doc.build(elements)
            st.success(f"Chat exported to {filename}")
            with open(filename, "rb") as f:
                st.download_button(
                    label="Download PDF",
                    data=f,
                    file_name=filename,
                    mime="application/pdf"
                )
