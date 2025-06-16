from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from Model_loader import llm


def build_chain(client):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(client=client, collection_name="enterprise_docs", embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )


# from rank_bm25 import BM25Okapi
# import numpy as np
#
#
# # 在一个单独函数中添加 BM25 检索
# def bm25_retrieval(processed_docs, query, k=1):
#     # 提取文档内容和元数据
#     doc_texts = [doc.page_content for doc in processed_docs]
#     doc_metadata = [doc.metadata for doc in processed_docs]
#
#     # 创建 BM25 模型并训练
#     tokenized_docs = [doc.split() for doc in doc_texts]
#     bm25 = BM25Okapi(tokenized_docs)
#
#     # 对查询进行分词
#     tokenized_query = query.split()
#
#     # 检索得分最高的文档
#     scores = bm25.get_scores(tokenized_query)
#     indices = np.argsort(scores)[::-1][:k]
#
#     # 返回得分最高的文档
#     retrieved_docs = []
#     for idx in indices:
#         retrieved_doc = Document(
#             page_content=doc_texts[idx],
#             metadata=doc_metadata[idx]
#         )
#         retrieved_docs.append(retrieved_doc)
#
#     return retrieved_docs
#
#
# # 在 build_chain 函数中整合 BM25 和向量检索
# def build_chain(client):
#     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#     vectorstore = Chroma(client=client, collection_name="enterprise_docs", embedding_function=embeddings)
#     retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
#
#     # 获取所有文档用于 BM25
#     all_docs = vectorstore.similarity_search(query="dummy", k=len(vectorstore))
#
#     # 创建一个自定义的检索链，结合 BM25 和向量检索
#     def hybrid_retrieval(query):
#         # 向量检索
#         vector_results = retriever.get_relevant_documents(query)
#
#         # BM25 检索
#         bm25_results = bm25_retrieval(all_docs, query)
#
#         # 合并结果（可以加权或取并集）
#         combined_results = vector_results + bm25_results
#         return combined_results
#
#     # 使用自定义检索链构建问答系统
#     return RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=lambda query: hybrid_retrieval(query),
#         return_source_documents=True
