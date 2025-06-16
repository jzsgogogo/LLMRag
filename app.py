import os
import time

import streamlit as st
# 主入口函数
from Model_loader import llm
from chroma_utils import init_chroma_client, add_to_chroma
from document import load_document, process_documents
from qa_chain import build_chain


def main():
    st.title("企业文档问答 & 本地模型聊天系统")
    st.divider()

    # 初始化 session_state
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.docs_loaded = False
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = []

    # —— 1. 上传文件 & 处理逻辑 —— #
    uploaded_file = st.file_uploader("上传文档（支持 pdf/txt/json/md）以进入向量问答模式",
                                     type=["pdf", "txt", "json", "md"])
    if uploaded_file:
        with st.spinner("处理中..."):
            # 保存到本地
            file_path = os.path.join("docs", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            # 加载、切分、入库
            docs = load_document(file_path)
            docs_split, vectors = process_documents(docs)
            client = init_chroma_client()
            add_to_chroma(client, docs_split, vectors)

            # 记录并提示
            chunk_count = len(docs_split)
            st.session_state.processed_files.append((uploaded_file.name, chunk_count))
            st.success(f"{uploaded_file.name} 处理完成，共 {chunk_count} 个片段")
            st.session_state.docs_loaded = True

    # —— 2. 渲染所有处理过的文件 —— #
    if st.session_state.processed_files:
        st.markdown("### 已处理的文档列表")
        for fname, cnt in st.session_state.processed_files:
            st.markdown(f"- **{fname}**: {cnt} 个片段")

    # 渲染历史消息
    for msg in st.session_state.messages:
        message_container = st.container()

        with message_container:
            if msg['role'] == 'user':
                st.markdown(f"""
                    <div style="
                        background-color: #f0f0f0;
                        padding: 15px 20px;
                        margin: 5px 0;
                        border-radius: 10px;
                        max-width: 80%;
                        word-break: break-word;
                        display: inline-block;
                        float: right;
                    ">{msg['content']}</div>
                """, unsafe_allow_html=True)
            else:
                if "reasoning" in msg and "answer" in msg:
                    html = f"""
                    <div style="
                        background-color: #e6f7ff;
                        padding: 15px 20px;
                        margin: 5px 0;
                        border-radius: 10px;
                        max-width: 80%;
                        word-break: break-word;
                        display: inline-block;
                        float: left;
                    ">
                        <details>
                            <summary>推理过程</summary>
                            {msg['reasoning']}
                        </details>
                        <strong>答案：</strong> {msg['answer']}
                    </div>
                    """
                    st.markdown(html, unsafe_allow_html=True)
                elif "content" in msg:
                    st.markdown(f"""
                        <div style="
                            background-color: #e6f7ff;
                            padding: 15px 20px;
                            margin: 5px 0;
                            border-radius: 10px;
                            max-width: 80%;
                            word-break: break-word;
                            display: inline-block;
                            float: left;
                        ">{msg['content']}</div>
                    """, unsafe_allow_html=True)

    # 处理用户输入
    if prompt := st.chat_input("请输入问题或聊天内容："):
        # 添加用户消息到历史记录
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        # 创建新的消息容器
        new_container = st.container()

        # 显示用户消息
        with new_container:
            st.markdown(f"""
                <div style="
                    background-color: #f0f0f0;
                    padding: 15px 20px;
                    margin: 5px 0;
                    border-radius: 10px;
                    max-width: 80%;
                    word-break: break-word;
                    display: inline-block;
                    float: right;
                ">{prompt}</div>
            """, unsafe_allow_html=True)

        # 创建AI回复占位符
        ai_reasoning_placeholder = st.empty()  # 推理过程
        ai_answer_placeholder = st.empty()     # 答案

        # 生成模型回复
        client = init_chroma_client()
        if st.session_state.docs_loaded:
            qa = build_chain(client)
            #callback = StreamlitCallbackHandler(st.container())
            with st.spinner("正在检索并生成回答..."):
                result = qa({"query": prompt})
            full_response = result["result"]
        else:
            with st.spinner("正在生成回复..."):
                full_response = llm(prompt)

        # 尝试将推理过程和答案分开
        if "Helpful Answer" in full_response:
            reasoning, answer = full_response.split("Helpful Answer", 1)
            answer = answer.lstrip()  # 去掉前导空白
        else:
            reasoning = "这是推理过程..."
            answer = full_response

        # 逐字显示推理过程
        reasoning_display = ""
        for char in reasoning:
            reasoning_display += char
            ai_reasoning_placeholder.markdown(f"""
                <div style="
                    background-color: #e6f7ff;
                    padding: 15px 20px;
                    margin: 5px 0;
                    border-radius: 10px;
                    max-width: 80%;
                    word-break: break-word;
                    display: inline-block;
                    float: left;
                ">
                    <details>
                        <summary>推理过程</summary>
                        {reasoning_display}
                    </details>
                </div>
            """, unsafe_allow_html=True)
            time.sleep(0.05)  # 控制字符显示速度

        # 逐字显示答案
        answer_display = ""
        for char in answer:
            answer_display += char
            ai_answer_placeholder.markdown(f"""
                <div style="
                    background-color: #e6f7ff;
                    padding: 15px 20px;
                    margin: 5px 0;
                    border-radius: 10px;
                    max-width: 80%;
                    word-break: break-word;
                    display: inline-block;
                    float: left;
                ">
                    <strong>答案：</strong> {answer_display}
                </div>
            """, unsafe_allow_html=True)
            time.sleep(0.05)  # 控制字符显示速度

        # 将完整回复添加到消息历史
        st.session_state.messages.append({
            "role": "assistant",
            "reasoning": reasoning,
            "answer": answer
        })

if __name__ == "__main__":
    main()