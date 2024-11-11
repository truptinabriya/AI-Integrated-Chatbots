from flask import Flask, render_template, request, jsonify
from general_chatbot import generate_response  # Import general chatbot logic
from document_chatbot import process_question_and_files  # Import document chatbot logic

app = Flask(__name__)
app.secret_key = "supersecretkey"

@app.route('/')
def index():
    return render_template('index.html')  # Default landing page with options to toggle chatbots

@app.route('/document_chatbot')
def document_chatbot():
    return render_template('document_chatbot.html', chatbot_type="Document ChatBot")

@app.route('/general_chatbot')
def general_chatbot():
    return render_template('general_chatbot.html', chatbot_type="General ChatBot")

# General ChatBot API for AJAX requests
@app.route('/general_chat', methods=["POST"])
def general_chat():
    user_role = request.json.get("role", "")
    user_message = request.json.get("message", "")

    if not user_role:
        return jsonify({"error": "Please provide a role."}), 400
    
    bot_response = generate_response(user_role, user_message)
    return jsonify({"response": bot_response})

# Document ChatBot API for AJAX requests
@app.route('/document_chat', methods=["POST"])
def document_chat():
    user_question = request.form.get("question", "")
    files = request.files.getlist("file")

    if user_question and files:
        response = process_question_and_files(user_question, files)
        if response:
            return jsonify({"response": response})
        else:
            return jsonify({"error": "Failed to process the request. Please try again."})
    else:
        return jsonify({"error": "Please provide a valid question and upload files."})

if __name__ == '__main__':
    app.run(debug=True)
