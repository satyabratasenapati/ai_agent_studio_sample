from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI # Still needed for TypedDict hint, but LLM is initialized in create_agent
from typing import Dict, List, Union, TypedDict, Literal
# Import InvoiceData directly from extraction.py, which now uses Pydantic v2
from core.ai_patterns import create_extraction_skill, create_translation_skill, create_summarization_skill, InvoiceData
import json # Import json for manual serialization if needed

# --- Define the Agent's State ---
class InvoiceProcessingState(TypedDict):
    invoice_content: str
    extracted_data: Union[InvoiceData, str, None]
    translated_content: str = None
    validation_result: str = None
    summary: str = None
    language_detected: str = None
    needs_translation: bool = False
    openai_api_key: str # Add API key to the state to pass it to nodes if needed

# --- Define Nodes (Functions representing steps in the workflow) ---

def load_invoice_content(state: InvoiceProcessingState) -> Dict:
    print("--- Node: Loading Invoice Content ---")
    return {"invoice_content": state["invoice_content"], "openai_api_key": state["openai_api_key"]}

def extract_invoice_data_node(state: InvoiceProcessingState) -> Dict:
    print("--- Node: Extracting Invoice Data ---")
    extractor = create_extraction_skill(state["openai_api_key"])
    extracted = extractor(state["invoice_content"])
    return {"extracted_data": extracted}

def check_invoice_language_node(state: InvoiceProcessingState) -> Dict:
    print("--- Node: Checking Language ---")
    content = state["invoice_content"]
    language = "English"
    needs_translation = False

    if "Factura" in content or "Fecha" in content or "Total" in content.lower() and "€" in content:
        language = "Spanish"
        needs_translation = True
    elif "Rechnung" in content or "Datum" in content or "Gesamt" in content.lower() and "€" in content:
        language = "German"
        needs_translation = True

    print(f"Detected Language: {language}, Needs Translation: {needs_translation}")
    return {"language_detected": language, "needs_translation": needs_translation}

def translate_invoice_node(state: InvoiceProcessingState) -> Dict:
    print("--- Node: Translating Invoice ---")
    if state["needs_translation"]:
        translator = create_translation_skill(state["openai_api_key"], target_language="English")
        translated_result = translator(state["invoice_content"])
        return {"translated_content": translated_result["translated_text"]}
    return {"translated_content": state["invoice_content"]}

def validate_invoice_data_node(state: InvoiceProcessingState) -> Dict:
    print("--- Node: Validating Invoice Data ---")
    extracted_data = state["extracted_data"]
    mock_enterprise_db = {
        "INV-2024-001": {"total_amount": 250.00, "currency": "USD"},
        "FAC-2023-11-15": {"total_amount": 250.00, "currency": "EUR"},
        "RECH-2024-03-20": {"total_amount": 500.00, "currency": "EUR"},
    }

    validation_messages = []

    if isinstance(extracted_data, InvoiceData):
        invoice_number = extracted_data.invoice_number
        total_amount = extracted_data.total_amount
        currency = extracted_data.currency

        if invoice_number in mock_enterprise_db:
            expected_data = mock_enterprise_db[invoice_number]
            if abs(total_amount - expected_data["total_amount"]) > 0.01:
                validation_messages.append(
                    f"AMOUNT MISMATCH: Invoice {invoice_number} - Expected {expected_data['total_amount']} {expected_data['currency']}, Got {total_amount} {currency}."
                )
            if currency != expected_data["currency"]:
                validation_messages.append(
                    f"CURRENCY MISMATCH: Invoice {invoice_number} - Expected {expected_data['currency']}, Got {currency}."
                )
        else:
            validation_messages.append(f"INVOICE NOT FOUND: Invoice number {invoice_number} not in enterprise system.")
    else:
        validation_messages.append(f"EXTRACTION FAILED: Cannot validate due to prior extraction error: {extracted_data}")

    if not validation_messages:
        validation_result = "Validation SUCCESS: Invoice data matches enterprise records."
    else:
        validation_result = "Validation FAILED:\n" + "\n".join(validation_messages)

    print(f"Validation Result: {validation_result}")
    return {"validation_result": validation_result}

def summarize_process_node(state: InvoiceProcessingState) -> Dict:
    print("--- Node: Summarizing Process ---")
    summarizer = create_summarization_skill(state["openai_api_key"])

    # Convert InvoiceData object to a JSON string for better summarization input
    extracted_data_str = ""
    if isinstance(state['extracted_data'], InvoiceData):
        extracted_data_str = state['extracted_data'].model_dump_json(indent=2) # Use model_dump_json for Pydantic v2
    elif isinstance(state['extracted_data'], str):
        extracted_data_str = state['extracted_data'] # It's an error string
    else:
        extracted_data_str = "No extracted data available."

    text_to_summarize = (
        f"Invoice Content (Original):\n{state['invoice_content']}\n\n"
        f"Extracted Data:\n{extracted_data_str}\n\n" # Use the serialized string
        f"Language Detected: {state['language_detected']}\n"
        f"Needs Translation: {state['needs_translation']}\n"
    )
    if state["needs_translation"]:
        text_to_summarize += f"Translated Content:\n{state['translated_content']}\n\n"
    text_to_summarize += f"Validation Result:\n{state['validation_result']}\n"

    summary_result = summarizer(text_to_summarize)
    print(f"Final Summary: {summary_result['summary']}")
    return {"summary": summary_result["summary"]}

# --- Define Conditional Edges ---
def route_after_language_check(state: InvoiceProcessingState) -> Literal["translate", "validate"]:
    if state["needs_translation"]:
        return "translate"
    else:
        return "validate"

# --- Build the LangGraph Workflow ---
def create_invoice_processor_agent():
    workflow = StateGraph(InvoiceProcessingState)

    workflow.add_node("load_content", load_invoice_content)
    workflow.add_node("extract_data", extract_invoice_data_node)
    workflow.add_node("check_language", check_invoice_language_node)
    workflow.add_node("translate_content", translate_invoice_node)
    workflow.add_node("validate_data", validate_invoice_data_node)
    workflow.add_node("summarize_process", summarize_process_node)

    workflow.set_entry_point("load_content")

    workflow.add_edge("load_content", "extract_data")
    workflow.add_edge("extract_data", "check_language")

    workflow.add_conditional_edges(
        "check_language",
        route_after_language_check,
        {
            "translate": "translate_content",
            "validate": "validate_data",
        }
    )
    workflow.add_edge("translate_content", "validate_data")
    workflow.add_edge("validate_data", "summarize_process")

    workflow.add_edge("summarize_process", END)

    return workflow.compile()

invoice_agent = create_invoice_processor_agent()

# Example usage (for testing this module directly)
if __name__ == "__main__":
    TEST_API_KEY = "YOUR_OPENAI_API_KEY_HERE"
    if TEST_API_KEY == "YOUR_OPENAI_API_KEY_HERE":
        print("WARNING: Please replace 'YOUR_OPENAI_API_KEY_HERE' with your actual key for local testing.")
        exit()

    sample_invoice_en = """
    Invoice #INV-2024-001
    Date: 2024-05-29
    Customer: Global Tech Solutions
    Description      Qty  Unit Price  Amount
    --------------------------------------
    Software License 1    100.00      100.00
    Consulting Hours 5    30.00       150.00
    --------------------------------------
    TOTAL: $250.00 USD
    """

    print("--- Running English Invoice Process ---")
    result_en = invoice_agent.invoke({"invoice_content": sample_invoice_en, "openai_api_key": TEST_API_KEY})
    print("\nFinal State (English Invoice):\n", result_en)
