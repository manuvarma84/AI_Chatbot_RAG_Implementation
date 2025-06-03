import os
import pickle
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

VECTORSTORE_PATH = "vectorstore.pkl"

def build_rag_pipeline(vectorstore):
    retriever = vectorstore.as_retriever()

    # Load the Hugging Face model and tokenizer for text summarization
    tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
    model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")

    # Create a Hugging Face pipeline for summarization
    hf_pipeline = pipeline("summarization", model=model, tokenizer=tokenizer)

    # Integrate Hugging Face pipeline into LangChain
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    prompt = PromptTemplate(
        template="\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:",
        input_variables=["context", "question"],
    )

    qa_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

    return qa_chain

def query_rag(qa_chain, query):
    response = qa_chain.invoke({"query": query})  # Use `invoke` instead of `__call__`

    # Extract and display the result
    result = response["result"].strip()
    print("\nAnswer:", result)

    # Display source documents
    print("\nSource Documents:")
    for doc in response["source_documents"]:
        print(f"- Source: {doc.metadata['source']}")
        print(f"Content: {doc.page_content[:500]}...\n")  # Show first 500 characters of the document for brevity

if __name__ == "__main__":
    if not os.path.exists(VECTORSTORE_PATH):
        print("Vector store not found. Please run the indexing script first.")
        exit()

    print("Loading vector store from disk...")
    with open(VECTORSTORE_PATH, "rb") as f:
        vectorstore = pickle.load(f)

    rag_pipeline = build_rag_pipeline(vectorstore)

    while True:
        try:
            user_query = input("\nEnter your question (or type 'exit' to quit): ")
            if user_query.lower() == "exit":
                print("Exiting. Goodbye!")
                break
            query_rag(rag_pipeline, user_query)
        except KeyboardInterrupt:
            print("\nExiting. Goodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
