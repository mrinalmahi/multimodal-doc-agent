from .agent_tools import build_tools_for_documents
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()

BASE_SYSTEM_PROMPT = (
    "Keep your answers concise and direct -- answer only what was "
    "asked, without extra caveats, alternative interpretations, or "
    "additional information the user didn't ask for. If a follow-up "
    "question refers to something in the earlier conversation, use "
    "that context rather than asking the user to repeat themselves. "
    "If a question is genuinely ambiguous -- for example, it doesn't "
    "specify which document, table, figure, model, or dataset is "
    "meant, and this can't be inferred from the conversation so far "
    "-- ask a brief clarifying question instead of guessing or "
    "listing out every possible interpretation.\n\n"
    "Only state facts that are directly supported by what a tool "
    "actually returned in this conversation -- do not invent details "
    "no tool provided. However, if a tool's returned text clearly "
    "contains the answer, even as part of a longer sentence or a "
    "parenthetical detail, treat it as reliable and state it "
    "confidently -- do not second-guess or withhold information a "
    "tool has already clearly given you. Only say you don't have "
    "the information when no tool result actually addresses the "
    "question at all."
)


def build_agent(document_ids: list[str], document_labels: dict | None = None):
    document_labels = document_labels or {}

    if document_ids:
        doc_list_str = "\n".join(
            f"- document_id: \"{doc_id}\" -- {document_labels.get(doc_id, doc_id)}"
            for doc_id in document_ids
        )

        clarification_instruction = ""
        if len(document_ids) > 1:
            clarification_instruction = (
                "\nIMPORTANT: There is more than one document available. "
                "If the user's question does not specify (or make clear "
                "from earlier conversation context) which document they "
                "mean, you MUST ask which document they're referring to "
                "before answering. Do not default to the first document "
                "listed, and do not guess. This applies even to questions "
                "that sound general, like \"how many tables are there\" "
                "or \"what does this paper say about X\" -- always confirm "
                "which specific document is meant first.\n"
            )

        documents_section = (
            "You are answering questions about the following documents "
            "available in this conversation:\n" + doc_list_str + "\n\n"
            "Every tool call requires a document_id argument. Match the "
            "user's reference (a filename, title, or casual mention like "
            "\"the first paper\") to the correct document_id from the list "
            "above." + clarification_instruction + "\n"
        )
    else:
        documents_section = (
            "No documents have been uploaded yet. If the user asks a "
            "question, let them know they need to upload a document first.\n\n"
        )

    system_prompt = documents_section + BASE_SYSTEM_PROMPT

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    tools, used_images = build_tools_for_documents(document_ids)
    agent = create_agent(llm, tools, system_prompt=system_prompt)
    return agent, used_images


def extract_text(content):
    """
    Normalize LLM response content to plain text, whether it comes
    back as a plain string or a list of content blocks.
    """
    if isinstance(content, list):
        return "\n".join(
            block["text"] for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return content


if __name__ == "__main__":
    doc_ids = ["jcim_paper"]
    labels = {"jcim_paper": "Explainability Methods from Machine Learning..."}

    agent, used_images = build_agent(doc_ids, labels)
    question = "What was the MSE for GATConv in Table 3 on the KIBA cold-target split?"
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})

    final_content = result["messages"][-1].content
    print(extract_text(final_content))
    print("Used images:", used_images)