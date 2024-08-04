import streamlit as st
import firebase_admin
from firebase_admin import credentials, auth, db
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import datetime
import google.generativeai as genai
import pyrebase

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
firebase_api_key = os.getenv("FIREBASE_API_KEY")

# Firebase configuration for Pyrebase
firebase_config = {
    "apiKey": firebase_api_key,
    "authDomain": os.getenv("FIREBASE_AUTH_DOMAIN"),
    "databaseURL": os.getenv("FIREBASE_DATABASE_URL"),
    "projectId": os.getenv("FIREBASE_PROJECT_ID"),
    "storageBucket": os.getenv("FIREBASE_STORAGE_BUCKET"),
    "messagingSenderId": os.getenv("FIREBASE_MESSAGING_SENDER_ID"),
    "appId": os.getenv("FIREBASE_APP_ID"),
}

# Initialize Pyrebase
firebase = pyrebase.initialize_app(firebase_config)
auth_pyrebase = firebase.auth()

# Initialize Firebase Admin SDK
def initialize_firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate('firebase_service_account.json')
        firebase_admin.initialize_app(cred, {
            'databaseURL': os.getenv("FIREBASE_DATABASE_URL")
        })

initialize_firebase()

# Firebase Auth functions
def register_user(email, password):
    try:
        user = auth_pyrebase.create_user_with_email_and_password(email, password)
        return f"User created successfully: {user['localId']}"
    except Exception as e:
        return f"Error creating user: {str(e)}"

def authenticate_user(email, password):
    try:
        user = auth_pyrebase.sign_in_with_email_and_password(email, password)
        return user
    except Exception as e:
        return None

# Firebase Realtime Database functions
def log_question_answer(user_id, question, answer):
    try:
        ref = db.reference(f'question_answer_history/{user_id}')
        ref.push({
            'question': question,
            'answer': answer,
            'timestamp': datetime.datetime.now().isoformat()
        })
    except Exception as e:
        print(f"Error logging question and answer: {str(e)}")

def get_question_answer_history(user_id):
    try:
        ref = db.reference(f'question_answer_history/{user_id}')
        history = ref.order_by_child('timestamp').get()
        return history
    except Exception as e:
        print(f"Error retrieving question-answer history: {str(e)}")
        return {}

# PDF and LangChain Functions
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, user_id):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Attempt to load the FAISS index and handle errors
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    except ValueError as e:
        st.error(f"Error loading FAISS index: {e}")
        return

    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    
    # Generate and display the response
    try:
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        answer = response["output_text"]
        st.write("Reply: ", answer)
        log_question_answer(user_id, user_question, answer)
    except Exception as e:
        st.error(f"Error generating response: {e}")

def get_gemini_response(question):
    model = genai.GenerativeModel("gemini-pro")
    chat = model.start_chat(history=[])
    response = chat.send_message(question, stream=True)
    return response

# Streamlit UI
def main():
    st.set_page_config(page_title="Tech-Titans")

    # Define authentication session state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'register_done' not in st.session_state:
        st.session_state.register_done = False
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None

    if not st.session_state.authenticated:
        st.sidebar.title("Authentication")
        if st.session_state.register_done:
            menu = st.sidebar.selectbox("Select an option", ["Login"])
        else:
            menu = st.sidebar.selectbox("Select an option", ["Register", "Login"])

        if menu == "Register":
            st.header("Register")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            if st.button("Register"):
                result = register_user(email, password)
                st.write(result)
                if "User created successfully" in result:
                    st.session_state.register_done = True

        elif menu == "Login":
            st.header("Login")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            if st.button("Login"):
                user = authenticate_user(email, password)
                if user:
                    st.session_state.authenticated = True
                    st.session_state.user_id = user['localId']
                else:
                    st.error("Wrong Information provided please try again.")

    else:
        # Authenticated user sees this
        st.header("TECH-TITANS")

        # Show question-answer history
        if st.checkbox("Show Question-Answer History"):
            history = get_question_answer_history(st.session_state.user_id)
            if history:
                st.write("Question-Answer History:")
                for key, value in history.items():
                    st.write(f"Question: {value['question']}, Answer: {value['answer']}, Timestamp: {value['timestamp']}")
            else:
                st.write("No question-answer history found.")

        user_question = st.text_input("Ask a Question from the PDF Files")

        if user_question:
            user_input(user_question, st.session_state.user_id)

        with st.sidebar:
            st.title("Menu:")
            pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
            if st.button("Submit & Process"):
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done")

        # Chat interface with Generative AI
        st.subheader("Chat with AI")
        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []

        chat_input = st.text_input("Input: ", key="chat_input")
        submit_chat = st.button("Ask the question")

        if submit_chat and chat_input:
            response = get_gemini_response(chat_input)
            st.session_state['chat_history'].append(("You", chat_input))
            st.subheader("The Response is")
            response_text = ""
            for chunk in response:
                st.write(chunk.text)
                response_text += chunk.text
            st.session_state['chat_history'].append(("Bot", response_text))

            # Save chat history to Firebase
            log_question_answer(st.session_state.user_id, chat_input, response_text)

        # Collapsible chat history
        with st.expander("Chat History"):
            st.subheader("The Chat History is")
            for role, text in st.session_state.get('chat_history', []):
                st.write(f"{role}: {text}")

        # Button to save chat history
        if st.button("Save Chat History"):
            with open("chat_history.txt", "w") as file:
                for role, text in st.session_state.get('chat_history', []):
                    file.write(f"{role}: {text}\n")
            st.success("Chat history saved!")

            # Display a download link for the saved chat history
            with open("chat_history.txt", "rb") as file:
                st.download_button(
                    label="Download Chat History",
                    data=file,
                    file_name="chat_history.txt",
                    mime="text/plain",
                )

if __name__ == "__main__":
    main()
