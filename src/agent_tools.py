import os
import json
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import base64

load_dotenv()

SHARED_COLLECTION_NAME = "all_documents"


# ---- shared helpers ----

def _encode_image(image_path):
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    return base64.b64encode(image_bytes).decode("utf-8")


def _ask_gemini_about_image(image_path, question, model_name="gemini-3.5-flash-lite"):
    llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=os.getenv("GOOGLE_API_KEY"))
    image_base64 = _encode_image(image_path)

    plain_text_instruction = (
        "Answer in plain text only. Do not use LaTeX notation, markdown "
        "formatting, or special symbols for subscripts/superscripts."
    )
    full_question = f"{question}\n\n{plain_text_instruction}"

    message = HumanMessage(content=[
        {"type": "text", "text": full_question},
        {"type": "image_url", "image_url": f"data:image/png;base64,{image_base64}"}
    ])

    response = llm.invoke([message])
    if isinstance(response.content, list):
        return "\n".join(b["text"] for b in response.content if isinstance(b, dict) and b.get("type") == "text")
    return response.content


_shared_vector_store = None

def _get_shared_vector_store():
    global _shared_vector_store
    if _shared_vector_store is None:
        embedding_model = HuggingFaceEmbeddings(model_name="intfloat/e5-base-v2")
        _shared_vector_store = Chroma(
            collection_name=SHARED_COLLECTION_NAME,
            embedding_function=embedding_model,
            persist_directory="data/chroma_db"
        )
    return _shared_vector_store


def _load_page_map(document_id, kind):
    """
    kind should be "table" or "figure". Loads the corresponding
    page-map JSON file for a document, returning an empty dict if
    it doesn't exist for some reason.
    """
    map_path = Path(f"data/processed/{document_id}/{kind}_page_map.json")
    if not map_path.exists():
        return {}
    with open(map_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---- tool factory: builds tools that work across multiple documents ----

def build_tools_for_documents(document_ids: list[str]):
    """
    Build a set of tools that operate across one or more documents.
    document_id is a real argument the LLM must supply on every call.

    Also tracks which images were actually used during this turn, via
    the used_images list, returned alongside the tools.
    """
    used_images = []

    @tool
    def search_text(document_id: str, query: str) -> str:
        """
        Search a specific document's main text for relevant passages.
        document_id must be one of the documents available in this
        conversation. Use this for questions about methodology,
        definitions, explanations, findings described in prose, or
        any general question about what a document says -- NOT for
        reading specific numeric values from a table, interpreting a
        chart/diagram, or counting/listing how many tables or figures
        exist (use list_tables or list_figures for that instead).
        """
        if document_id not in document_ids:
            return f"Error: '{document_id}' is not one of the available documents."

        prefixed_query = f"query: {query}"
        vector_store = _get_shared_vector_store()
        results = vector_store.similarity_search_with_score(
            prefixed_query, k=10, filter={"document_id": document_id}
        )

        output_parts = []
        for doc, score in results:
            text = doc.metadata.get("original_text", doc.page_content)
            page = doc.metadata["page_number"]
            output_parts.append(f"[Page {page}]: {text}")

        return "\n\n".join(output_parts)

    @tool
    def list_tables(document_id: str) -> str:
        """
        List every table number that exists in a specific document.
        Use this to answer questions like "how many tables are
        there", "is there a Table N", or "what tables exist in this
        document" -- this is authoritative and should always be used
        for these questions instead of search_text, since it reads
        directly from the document's actual table index rather than
        guessing from retrieved text.
        """
        if document_id not in document_ids:
            return f"Error: '{document_id}' is not one of the available documents."

        table_map = _load_page_map(document_id, "table")
        if not table_map:
            return "This document has no detected tables."

        table_numbers = sorted(int(k) for k in table_map.keys())
        return f"This document has {len(table_numbers)} table(s): Table {', Table '.join(str(n) for n in table_numbers)}."

    @tool
    def list_figures(document_id: str) -> str:
        """
        List every figure number that exists in a specific document.
        Use this to answer questions like "how many figures are
        there", "is there a Figure N", or "what figures exist in this
        document" -- this is authoritative and should always be used
        for these questions instead of search_text.
        """
        if document_id not in document_ids:
            return f"Error: '{document_id}' is not one of the available documents."

        figure_map = _load_page_map(document_id, "figure")
        if not figure_map:
            return "This document has no detected figures."

        figure_numbers = sorted(int(k) for k in figure_map.keys())
        return f"This document has {len(figure_numbers)} figure(s): Figure {', Figure '.join(str(n) for n in figure_numbers)}."

    @tool
    def ask_about_table(document_id: str, table_number: int, question: str) -> str:
        """
        Answer a question about a specific numbered table in a
        specific document by visually reading its image. document_id
        must be one of the documents available in this conversation.
        Use this when the question asks for a specific numeric value
        (like MSE, RMSE, CI, or Pearson score) from a named table, or
        when the user asks to see/show a specific table.
        """
        if document_id not in document_ids:
            return f"Error: '{document_id}' is not one of the available documents."

        image_path = Path(f"data/processed/{document_id}/tables") / f"table_{table_number}.png"
        if not image_path.exists():
            return f"Error: no image found for Table {table_number} in document '{document_id}'"

        used_images.append({"path": str(image_path), "caption": f"Table {table_number}"})

        return _ask_gemini_about_image(str(image_path), question)

    @tool
    def ask_about_figure(document_id: str, figure_number: int, question: str) -> str:
        """
        Answer a question about a specific numbered figure (chart,
        diagram, or plot) in a specific document by visually reading
        the page it appears on. document_id must be one of the
        documents available in this conversation. Use this for
        questions about what a figure shows, describes, or
        illustrates, or when the user asks to see/show a specific
        figure.
        """
        if document_id not in document_ids:
            return f"Error: '{document_id}' is not one of the available documents."

        figure_page_map = _load_page_map(document_id, "figure")
        page_nums = figure_page_map.get(str(figure_number))
        if not page_nums:
            return f"Error: no page found for Figure {figure_number} in document '{document_id}'"

        page_num = page_nums[0]
        image_path = Path(f"data/processed/{document_id}/figures") / f"page_{page_num}.png"
        if not image_path.exists():
            return f"Error: no image found for page {page_num}"

        used_images.append({"path": str(image_path), "caption": f"Figure {figure_number}"})

        return _ask_gemini_about_image(str(image_path), question)

    tools = [search_text, list_tables, list_figures, ask_about_table, ask_about_figure]
    return tools, used_images