from flask import Flask, request, jsonify
from flask_cors import CORS
import os # Still import os, but no longer checking for environment variable
from core.agents.invoice_processor import invoice_agent, InvoiceProcessingState

app = Flask(__name__)
CORS(app)

@app.route('/process_invoice', methods=['POST'])
def process_invoice():
    """
    API endpoint to receive invoice content and process it using the LangGraph agent.
    Now expects 'invoice_content' AND 'openai_api_key' in the JSON payload.
    """
    data = request.get_json()
    invoice_content = data.get('invoice_content')
    openai_api_key = data.get('openai_api_key') # Get API key from request body

    # Basic input validation
    if not invoice_content:
        return jsonify({"error": "No 'invoice_content' provided in the request body."}), 400
    if not openai_api_key:
        return jsonify({"error": "No 'openai_api_key' provided in the request body. Please enter your OpenAI API Key."}), 400

    try:
        # Initialize the LangGraph agent's state with both invoice content and the API key.
        initial_state: InvoiceProcessingState = {
            "invoice_content": invoice_content,
            "openai_api_key": openai_api_key, # Pass the API key in the initial state
            "extracted_data": None,
            "translated_content": None,
            "validation_result": None,
            "summary": None,
            "language_detected": None,
            "needs_translation": False
        }

        # Invoke the LangGraph agent with the initial state.
        final_state = invoice_agent.invoke(initial_state)

        # Return the final state of the agent as a JSON response.
        return jsonify(final_state), 200
    except Exception as e:
        print(f"Error processing invoice: {e}")
        return jsonify({"error": str(e), "message": "An internal server error occurred during invoice processing."}), 500

if __name__ == '__main__':
    # No longer checking for environment variable here.
    # The API key is expected to come from the frontend request.
    print("Starting Flask app. Ensure you provide your OpenAI API Key in the frontend.")
    app.run(debug=True, port=5000)
