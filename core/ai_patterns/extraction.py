from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from typing import List, Dict, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the schema for invoice data extraction using Pydantic v2
class InvoiceData(BaseModel):
    invoice_number: str = Field(description="The unique invoice identification number.")
    invoice_date: str = Field(description="The date the invoice was issued (e.g.,YYYY-MM-DD).")
    total_amount: float = Field(description="The total amount due on the invoice.")
    currency: str = Field(description="The currency of the total amount (e.g., USD, EUR).")
    line_items: List[Dict[str, Union[str, float]]] = Field(
        description="A list of dictionaries, each representing a line item with 'description', 'quantity', and 'unit_price'."
    )

# Create a parser for our Pydantic model. PydanticOutputParser should handle v2 models.
extraction_parser = PydanticOutputParser(pydantic_object=InvoiceData)

def create_extraction_skill(openai_api_key: str):
    """
    Creates an extraction skill that uses an LLM to parse invoice data into a structured format.
    The OpenAI API key is passed directly here.
    """
    from langchain.prompts import ChatPromptTemplate
    # Initialize ChatOpenAI with the provided API key
    try:
        model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)
    except Exception as e:
        logger.error(f"Error initializing ChatOpenAI: {e}")
        def failing_extract(document_content: str):
            return f"Error initializing LLM for extraction: {e}"
        return failing_extract

    prompt_template = """You are an expert at extracting structured information from documents.
    Extract the following details from the provided invoice content:

    {document_content}

    {format_instructions}

    If a field is not found, use a suitable placeholder like 'N/A' for strings or 0.0 for numbers,
    but try your best to extract accurate information.
    """
    prompt = ChatPromptTemplate.from_template(template=prompt_template)

    def extract(document_content: str) -> Union[InvoiceData, str]:
        """
        Executes the extraction process using the configured LLM and parser.
        """
        _input = prompt.format_prompt(
            document_content=document_content,
            format_instructions=extraction_parser.get_format_instructions()
        )
        output = None
        try:
            # Invoke the LLM with the formatted prompt
            output = model.invoke(_input)
            # Attempt to parse the LLM's output into our Pydantic model
            return extraction_parser.parse(output.content)
        except Exception as e:
            logger.error(f"Error during LLM invoke or parsing: {e}")
            raw_output = f" Raw LLM Output: {output.content}" if output else ""
            return f"Extraction Error: {e}.{raw_output}"

    return extract

# Example usage (for testing this module directly)
if __name__ == "__main__":
    # IMPORTANT: Replace with your actual OpenAI API key for testing
    TEST_API_KEY = "YOUR_OPENAI_API_KEY_HERE"
    if TEST_API_KEY == "YOUR_OPENAI_API_KEY_HERE":
        print("WARNING: Please replace 'YOUR_OPENAI_API_KEY_HERE' with your actual key for local testing.")
        exit()

    extractor = create_extraction_skill(TEST_API_KEY)
    if callable(extractor):
        sample_invoice = """
        INVOICE #INV-2024-001
        Date: 2024-05-29
        Customer: ABC Corp
        Description      Qty  Unit Price  Amount
        --------------------------------------
        Product A        2    50.00       100.00
        Service B        1    150.00      150.00
        --------------------------------------
        TOTAL: $250.00 USD
        """
        extracted_data = extractor(sample_invoice)
        print("Extracted Data:", extracted_data.model_dump_json(indent=2) if isinstance(extracted_data, InvoiceData) else extracted_data)
    else:
        print(extractor) # Print the error message from LLM initialization