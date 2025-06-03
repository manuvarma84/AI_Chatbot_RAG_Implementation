import os
from flask import Flask, request, jsonify, render_template, send_file, send_from_directory
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers import AutoModelForSeq2SeqLM
import spacy
from textblob import TextBlob  # To handle typos and spelling correction
import Levenshtein  # For partial matching
import numpy as np  # For cosine similarity
import torch    

hf_api_key = "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"

# Initialize Flask app
app = Flask(__name__)

# Directories for vectorstore and documents
VECTORSTORE_DIR = <<provide the path of the vectorstore here>>
DOCUMENTS_DIR = <<provide the path of the vectorstore here>>
DOCUMENTS_DIR_2 = <<provide the path of the documents folder here>>

# Initialize embeddings
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

# Load spaCy model for NLP processing
nlp = spacy.load("en_core_web_sm")


def correct_spelling(text):
    """Correct spelling in the input text using TextBlob."""
    blob = TextBlob(text)
    return str(blob.correct())


def lemmatize_and_extract_content_words(text):
    """Lemmatize and extract content words for search optimization."""
    doc = nlp(text)
    content_words = [
        token.lemma_ for token in doc
        if token.is_alpha and not token.is_stop
    ]
    return content_words


def compute_semantic_similarity(query_vector, doc_vector):
    """Compute cosine similarity between two vectors."""
    return np.dot(query_vector, doc_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(doc_vector))


def is_partial_match(query_word, target_word):
    """Check if two words are a partial match using Levenshtein similarity."""
    ratio = Levenshtein.ratio(query_word, target_word)
    return ratio >= 0.7


def build_rag_pipeline(vectorstore):
    """Build the Retrieval-Augmented Generation (RAG) pipeline."""
    retriever = vectorstore.as_retriever()

    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_api_key)
    #model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_api_key,  # optionally set to torch.float16 for performance if GPU is used
        device_map="auto",    # automatically map model to available GPU(s)
        torch_dtype=torch.float16
    )
    # Use a generation pipeline instead of summarization
    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.95,
        do_sample=True
    )

    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    #prompt = PromptTemplate(
    #    template="\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:",
    #    input_variables=["context", "question"],
    #)
    prompt = PromptTemplate(
        template="<s>[INST] Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer: [/INST]",
        input_variables=["context", "question"],
    )    

    #qa_chain = RetrievalQA.from_chain_type(
    #    llm, chain_type="stuff", retriever=retriever, return_source_documents=True
    #)
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )    

    return qa_chain


def query_rag_with_more_info(qa_chain, query, line_limit=3):
    """Query the RAG pipeline and provide truncated content with 'More Info'."""
    response = qa_chain.invoke({"query": query})
    result = response["result"].strip()
    source_documents = response.get("source_documents", [])

    sources = {}
    stop_words = set([
        "the", "in", "if", "what", "where", "when", "whether", "can", "be",
        "how", "why", "and", "or", "is", "of", "to", "for", "on", "with"
    ])

    for doc in source_documents:
        file_name = os.path.basename(doc.metadata.get("source", "Unknown"))
        content = doc.page_content.strip()
        query_words = lemmatize_and_extract_content_words(query)

        matching_content = []
        for query_word in query_words:
            for word in content.split():
                if is_partial_match(query_word, word):
                    matching_content.append(word)

        query_vector = embeddings.embed_query(query)
        doc_vector = embeddings.embed_query(content)
        similarity_score = compute_semantic_similarity(query_vector, doc_vector)

        non_stop_word_matches = [word for word in matching_content if word not in stop_words]

        if len(non_stop_word_matches) > 0 or similarity_score > 0.7:
            if file_name not in sources:
                sources[file_name] = []

            # Split content into lines
            content_lines = content.split("\n")
            truncated_content = "\n".join(content_lines[:line_limit])
            has_more_info = len(content_lines) > line_limit
            more_info_content = "\n".join(content_lines[line_limit:]) if has_more_info else ""

            sources[file_name].append({
                "content": truncated_content,
                "matching_words": matching_content,
                "similarity": similarity_score,
                "has_more_info": has_more_info,
                "more_info_content": more_info_content,
            })

    detailed_response = {
        "answer": result,
        "sources": [
            {
                "source": source,
                "contents": [
                    {
                        "content": content["content"],
                        "matching_words": content["matching_words"],
                        "similarity": content["similarity"],
                        "index": idx + 1,
                        "has_more_info": content["has_more_info"],
                    }
                    for idx, content in enumerate(contents)
                ],
            }
            for source, contents in sources.items()
        ],
        "more_info_data": [
            {
                "source": source,
                "more_info_content": content["more_info_content"],
                "index": idx + 1,
            }
            for source, contents in sources.items()
            for idx, content in enumerate(contents) if content["has_more_info"]
        ],
    }

    return detailed_response


@app.get("/")
def home():
    return render_template("base.html")


@app.post('/query')
def handle_query():
    try:
        data = request.json
        user_query = data.get('query')

        if not user_query:
            return jsonify({"error": "No query provided"}), 400

        corrected_query = correct_spelling(user_query)
        content_words = lemmatize_and_extract_content_words(corrected_query)

        if not content_words:
            return jsonify({"error": "Query contains no significant content"}), 400

        filtered_query = " ".join(content_words)

        if not os.path.exists(os.path.join(VECTORSTORE_DIR, "index.faiss")):
            return jsonify({"error": "Vectorstore files not found. Please run the indexing script first."}), 404

        vectorstore = FAISS.load_local(
            VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True
        )

        rag_pipeline = build_rag_pipeline(vectorstore)
        response = query_rag_with_more_info(rag_pipeline, filtered_query)

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
