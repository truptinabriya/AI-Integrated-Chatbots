
import os
from flask import Flask, render_template, request, flash
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS as CommunityFAISS
import docx2txt
from langchain.prompts import PromptTemplate
import PyPDF2
import csv

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

if google_api_key:
    genai.configure(api_key=google_api_key)

app = Flask(__name__)
app.secret_key = "supersecretkey"  # For flashing error messages

# Helper Function to Split Text into Chunks
def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(text)

# Helper Function to Handle File Processing
def extract_text_from_file(file):
    try:
        if file.filename.endswith(".txt"):
            return file.read().decode("utf-8")
        elif file.filename.endswith(".pdf"):
            return extract_text_from_pdf(file)
        elif file.filename.endswith(".docx"):
            return docx2txt.process(file)
        elif file.filename.endswith(".csv"):
            return extract_text_from_csv(file)
        else:
            flash(f"Unsupported file format: {file.filename}")
            return ""
    except Exception as e:
        flash(f"Error processing file {file.filename}: {str(e)}")
        return ""

# PDF Processing
def extract_text_from_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        return " ".join([page.extract_text() for page in pdf_reader.pages])
    except Exception as e:
        flash(f"Error extracting text from PDF: {str(e)}")
        return ""

# CSV Processing
def extract_text_from_csv(file):
    try:
        csv_file = file.read().decode("utf-8").splitlines()
        csv_reader = csv.reader(csv_file)
        return " ".join([" ".join(row) for row in csv_reader])
    except Exception as e:
        flash(f"Error extracting text from CSV: {str(e)}")
        return ""

# Create Vector Store with Embeddings
def create_vector_store(text_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = CommunityFAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local(f"faiss_index_gemini")
        return vector_store
    except Exception as e:
        flash(f"Error creating vector store: {str(e)}")
        return None

# Load FAISS Vector Store
def load_vector_store():
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        return CommunityFAISS.load_local(f"faiss_index_gemini", embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        flash(f"Error loading vector store: {str(e)}")
        return None

def get_conversational_chain():
    prompt_template = """
    Task: Imagine you are an Insurance or Legal expert who can read, understand, and extract data from legal contracts.
    If you don't know the answer, just say you don't know. DO NOT try to make up an answer.
    If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.
    Retain the original information as closely as possible; refrain from altering it.
    If you don’t receive any context say politely that the provided text does not contain any information about the question.
    Don't give me information that is not mentioned in the context.\n\n
    Context:\n {context}\n
    Question: \n{question}\n
    Answer:
    """

    # Correctly initialize the PromptTemplate with input variables
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # Load the conversational chain with the prompt template
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)

    return chain

# Handle Form Submission and Question Processing
def process_question_and_files(user_question, files):
    extracted_texts = [extract_text_from_file(file) for file in files]

    if any(extracted_texts):
        all_chunks = []
        for text in extracted_texts:
            all_chunks.extend(split_text_into_chunks(text))

        # Create the vector store with the processed text
        vector_store = create_vector_store(all_chunks)
        if not vector_store:
            return None

        # Load the FAISS index
        faiss_db = load_vector_store()
        if not faiss_db:
            return None

        # Run the conversational chain
        docs = faiss_db.similarity_search(user_question)
        chain = get_conversational_chain()
        if not chain:
            return None

        try:
            response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)["output_text"]
            return response
        except Exception as e:
            flash(f"Error generating response: {str(e)}")
            return None
    else:
        flash("No valid text extracted from uploaded files.")
        return None

# Home Route
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_question = request.form.get("question", "")
        files = request.files.getlist("file")

        if user_question and files:
            response = process_question_and_files(user_question, files)
            if response:
                return render_template("response.html", question=user_question, response=response)
            else:
                flash("Failed to process the request. Please try again.")
        else:
            flash("Please provide a valid question and upload files.")

    return render_template("index.html")

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
