

# 本地模型路径
import torch
from langchain_huggingface import HuggingFacePipeline
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM, pipeline
import streamlit as st
MODEL_PATH = "D:\models\deepseek-aiDeepSeek-R1-Distill-Qwen-7B"  # 你本地模型路径
CHROMA_PERSIST_DIR = "./chroma_db"

# ============ 加载本地模型并处理 GPU / CPU fallback =============
@st.cache_resource
def load_local_model():
    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            quantization_config=bnb_config,
            device_map="auto",
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        st.success("模型已加载到 GPU（4bit 量化）")
    except Exception as e:
        st.warning(f"GPU加载失败，已切换至CPU: {e}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            device_map={"": "cpu"},
            torch_dtype=torch.float32
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
    return HuggingFacePipeline(pipeline=pipe)


# 加载一次模型
llm = load_local_model()