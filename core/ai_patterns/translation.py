from langchain_openai import ChatOpenAI
from typing import Dict

def create_translation_skill(openai_api_key: str, target_language: str = "English"):
    """
    Creates a translation skill that translates text to a specified target language.
    The OpenAI API key is passed directly here.
    """
    from langchain.prompts import ChatPromptTemplate
    # Initialize ChatOpenAI with the provided API key
    model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)

    prompt_template = """Translate the following text to {target_language}.
    Only provide the translated text, do not add any conversational phrases.

    Text to translate:
    {text_to_translate}
    """
    prompt = ChatPromptTemplate.from_template(template=prompt_template)

    def translate(text_to_translate: str) -> Dict[str, str]:
        """
        Executes the translation process using the configured LLM.
        """
        _input = prompt.format_prompt(text_to_translate=text_to_translate, target_language=target_language)
        output = model.invoke(_input)
        return {"translated_text": output.content}

    return translate

# Example usage (for testing this module directly)
if __name__ == "__main__":
    # IMPORTANT: Replace with your actual OpenAI API key for testing
    TEST_API_KEY = "YOUR_OPENAI_API_KEY_HERE"
    if TEST_API_KEY == "YOUR_OPENAI_API_KEY_HERE":
        print("WARNING: Please replace 'YOUR_OPENAI_API_KEY_HERE' with your actual key for local testing.")
        exit()

    translator_es = create_translation_skill(TEST_API_KEY, target_language="Spanish")
    sample_text_en = "Hello, how are you doing today?"
    translated_es = translator_es(sample_text_en)
    print("Translated to Spanish:", translated_es)
