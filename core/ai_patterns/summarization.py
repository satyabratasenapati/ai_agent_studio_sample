from langchain_openai import ChatOpenAI
from typing import Dict

def create_summarization_skill(openai_api_key: str):
    """
    Creates a summarization skill that generates a concise summary of provided text.
    The OpenAI API key is passed directly here.
    """
    from langchain.prompts import ChatPromptTemplate
    # Initialize ChatOpenAI with the provided API key
    model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)

    prompt_template = """Summarize the following text concisely and accurately.
    Focus on the main points and key information.

    Text to summarize:
    {text_to_summarize}
    """
    prompt = ChatPromptTemplate.from_template(template=prompt_template)

    def summarize(text_to_summarize: str) -> Dict[str, str]:
        """
        Executes the summarization process using the configured LLM.
        """
        _input = prompt.format_prompt(text_to_summarize=text_to_summarize)
        output = model.invoke(_input)
        return {"summary": output.content}

    return summarize

# Example usage (for testing this module directly)
if __name__ == "__main__":
    # IMPORTANT: Replace with your actual OpenAI API key for testing
    TEST_API_KEY = "YOUR_OPENAI_API_KEY_HERE"
    if TEST_API_KEY == "YOUR_OPENAI_API_KEY_HERE":
        print("WARNING: Please replace 'YOUR_OPENAI_API_KEY_HERE' with your actual key for local testing.")
        exit()

    summarizer = create_summarization_skill(TEST_API_KEY)
    long_text = """
    The annual report for the fiscal year 2023 showed significant growth across all key metrics.
    Revenue increased by 15%, driven primarily by strong performance in the North American market
    and the successful launch of three new product lines. Operating expenses were well-managed,
    leading to a 20% increase in net profit. The company also invested heavily in research and
    development, with several promising projects in the pipeline for 2024. Employee satisfaction
    surveys indicated a positive work environment, and retention rates remained high. The board
    expressed confidence in the leadership team's strategy for continued expansion and innovation.
    """
    summary_result = summarizer(long_text)
    print("Summary:", summary_result)
