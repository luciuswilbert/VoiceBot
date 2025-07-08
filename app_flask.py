import os
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import fitz  # PyMuPDF
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.schema import SystemMessage, HumanMessage, Document
from google.generativeai import configure, GenerativeModel, upload_file
import tempfile

load_dotenv()
configure(api_key=os.getenv("GEMINI_API_KEY"))

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

faiss_db = None

# Helper functions (reuse from app.py)
def get_azure_embeddings():
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    embedding_deployment = os.getenv("EMBEDDING_DEPLOYMENT_NAME")
    embedding_model = os.getenv("EMBEDDING_MODEL_NAME")
    api_version = os.getenv("API_VERSION")
    return AzureOpenAIEmbeddings(
        azure_deployment=embedding_deployment,
        openai_api_key=azure_api_key,
        azure_endpoint=azure_endpoint,
        api_version=api_version,
        chunk_size=1
    )

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

def chunk_text(text, chunk_size=300, overlap=50):
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def transcribe_audio_with_gemini(audio_file_path):
    file = upload_file(audio_file_path)
    model = GenerativeModel(model_name="models/gemini-2.5-flash")
    response = model.generate_content([
        "Give me the transcription of this audio clip.",
        file
    ])
    return response.text

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    global faiss_db
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    if filename.lower().endswith('.pdf'):
        extracted_text = extract_text_from_pdf(file_path)
        chunks = chunk_text(extracted_text)
        embedding_fn = get_azure_embeddings()
        documents = [Document(page_content=chunk) for chunk in chunks]
        faiss_db = FAISS.from_documents(
            documents=documents,
            embedding=embedding_fn
        )
        faiss_db.save_local("my_faiss_index")
        return jsonify({"message": "PDF processed. You can now ask questions."})
    elif filename.lower().endswith('.mp3') or filename.lower().endswith('.wav'):
        transcription = transcribe_audio_with_gemini(file_path)
        return jsonify({"transcription": transcription})
    else:
        return jsonify({"error": "Unsupported file type."}), 400

@app.route("/ask", methods=["POST"])
def ask():
    global faiss_db
    data = request.get_json()
    user_query = data.get("question", "")
    if not user_query:
        return jsonify({"error": "No question provided."}), 400
    if faiss_db is None:
        # Try to load from disk
        if os.path.exists("my_faiss_index"):
            embedding_fn = get_azure_embeddings()
            faiss_db = FAISS.load_local(
                "my_faiss_index",
                embeddings=embedding_fn,
                allow_dangerous_deserialization=True
            )
        else:
            return jsonify({"error": "No knowledge base available. Please upload a PDF first."}), 400
    results = faiss_db.similarity_search(user_query, k=4)
    context = "\n\n".join([doc.page_content for doc in results])
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_deployment = os.getenv("DEPLOYMENT_NAME")
    api_version = os.getenv("API_VERSION")
    llm = AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        openai_api_key=azure_api_key,
        deployment_name=azure_deployment,
        api_version=api_version,
        temperature=0.1,
        streaming=False
    )
    system = SystemMessage(content="You are AI Assistant. Provide clear, accurate, and concise answers strictly based on the context provided. Ensure your responses are balanced in length—neither too brief nor overly detailed—delivering essential information effectively and efficiently. Avoid including any information not supported by the given context.")
    user = HumanMessage(content=f"Context:\n{context}\n\nUser Question: {user_query}\n\nAnswer using only the given context.")
    response = llm.invoke([system, user])
    return jsonify({"answer": response.content.strip()})

if __name__ == "__main__":
    app.run(debug=True) 