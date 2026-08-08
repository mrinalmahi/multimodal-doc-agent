from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def load_vector_store(collection_name="jcim_paper", db_path="data/chroma_db"):
    embedding_model = HuggingFaceEmbeddings(model_name="intfloat/e5-base-v2")
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=db_path
    )
    return vector_store


def search_text(query, vector_store, top_k=3):
    results = vector_store.similarity_search_with_score(query, k=top_k)

    output = []
    for doc, score in results:
        output.append({
            "text": doc.page_content,
            "page_number": doc.metadata["page_number"],
            "distance": score
        })

    return output


if __name__ == "__main__":
    vector_store = load_vector_store()

    query = "What optimizer was used for training the models?"
    results = search_text(query, vector_store, top_k=3)

    print(f"Query: {query}\n")
    for i, r in enumerate(results):
        print(f"Result {i+1} (page {r['page_number']}, distance {r['distance']:.4f}):")
        print(r["text"])
        print()