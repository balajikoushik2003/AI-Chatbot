from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import pipeline  # Import the transformers library
import torch

app = Flask(__name__)
CORS(app)

# Initialize the conversational pipeline with a pre-trained model.
# This model is used to generate responses in the chat.
# You can experiment with different models from Hugging Face Transformers.
# Some popular choices include:
# - "microsoft/DialoGPT-medium" (good for general conversation, relatively small)
# - "facebook/blenderbot-400M-distill" (more coherent, but larger)
# - "google/dialogpt-large"
# If you encounter errors, ensure you have enough RAM and try a smaller model.
conversational_pipeline = None  # Initialize to None *outside* any function

def load_pipeline():
    """Loads the conversational pipeline.  This is in a separate function
    to improve error handling and modularity."""
    global conversational_pipeline  # Declare that we're using the global variable
    try:
        # Check if CUDA is available and use it if so
        device = "cuda" if torch.cuda.is_available() else "cpu"
        conversational_pipeline = pipeline(
            "conversational", model="microsoft/DialoGPT-medium", device=device
        )
        print(f"Conversational pipeline loaded successfully. Using device: {device}")  # Add a success message
    except Exception as e:
        print(f"Error loading conversational pipeline: {e}")
        conversational_pipeline = None  # Ensure it's set to None on failure
        #  Consider logging the error message for debugging
        #  You might also want to try a different model here, or
        #  implement a retry mechanism.



# Load the pipeline when the app starts.  Crucially, call this *before*
# any route handlers that use conversational_pipeline.
load_pipeline()


# Store conversation history.  This is a *very* basic way to manage history,
# and is not suitable for production.  In a real application, you would use
# a database or a proper session management system.
#
# The history is stored as a dictionary where:
# - The key is a user identifier (in this case, a simple string).
# - The value is a transformers Conversation object, which holds the
#   sequence of messages in the conversation.
conversation_history = {}

@app.route("/", methods=["GET"])
def index():
    """
    This route serves the main HTML page for the chat application.
    It assumes you have a file named "AI.html" in a folder named "templates"
    in the same directory as this Python script.
    """
    return render_template("AI.html")  #  Make sure AI.html is in the templates folder


@app.route("/chat", methods=["POST", "OPTIONS"])
def chat():
    """
    This route handles incoming chat messages from the user.  It receives the
    user's message, generates a response using the conversational AI model,
    and sends the response back to the user.
    """
    if request.method == "OPTIONS":
        #  Handle preflight requests (OPTIONS requests) that browsers sometimes send
        #  before POST requests, especially with CORS.
        return jsonify({}), 200

    # Get the data sent from the client.  We expect the user's message
    # to be in the "messages" list within the JSON data.
    data = request.get_json(force=True)

    # Get a user identifier.  In a real application, this would come from
    # a proper authentication/session management system.  For this simple
    # example, we're using a header, and defaulting to "default_user" if
    # no header is provided.  This is NOT secure for real apps.
    user_id = request.headers.get('User-ID', 'default_user')

    # Extract the user's message.  We assume the *last* message in the
    # "messages" list is the user's current input.  This is important
    # for maintaining context.
    user_message = data.get("messages", [])[-1].get("content", "") if data.get("messages") else ""

    # Check if a message was provided.
    if not user_message:
        return jsonify({"error": "Please provide a message"}), 400

    try:
        if conversational_pipeline: # Check if the pipeline loaded successfully
            # Get the conversation history for this user.  If the user
            # doesn't have a history yet, create a new conversation.
            history = conversation_history.get(
                user_id, conversational_pipeline.create_new_conversation()
            )

            # Add the user's message to the conversation history.  This
            # tells the model what the user said.
            history.add_user_message(user_message)

            # Generate a response from the AI.  We pass the *entire*
            # conversation history to the model, so it can maintain context.
            response = conversational_pipeline(history)

            # Extract the AI's response from the output.  The pipeline
            # returns a Conversation object, which contains the generated
            # responses.  We want the *last* one.
            ai_response = response.generated_responses[-1]

            # Update the conversation history with the AI's response.
            # This is crucial for maintaining context in subsequent turns
            # of the conversation.
            history.add_ai_response(ai_response)
            conversation_history[user_id] = history  # Store the updated history

            # Return the AI's response to the client.
            return jsonify({"response": ai_response})
        else:
            # If the conversational pipeline failed to load (e.g., due to
            # an error during initialization), return a simple echo response.
            # This ensures the chat doesn't completely break.
            return jsonify(
                {"response": f"Echo: {user_message} (Conversational AI not loaded)"}
            )

    except Exception as e:
        # Handle any errors that occur during the chat processing (e.g.,
        # errors with the conversational model).  This is important for
        # robustness.
        print(f"Error during chat processing: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Start the Flask development server.  The 'debug=True' option
    # makes the server automatically reload when you make changes to the
    # code, which is helpful for development.  **Do not use 'debug=True'
    # in a production environment.**
    app.run(debug=True)
