import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings  # Updated import
from langchain_community.vectorstores import FAISS  # Updated import
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI  # Use ChatOpenAI for chat models
import os
import pickle
import openai
import tiktoken  # For token counting
import re
import pandas as pd
from io import BytesIO

# Sidebar for application description
with st.sidebar:
    st.title("PDF Simplifier")
    st.markdown('''
        ## About
        This app uses an LLM-powered chatbot built using:
       
        - [OpenAI](https://platform.openai.com/docs/models)  
        - [Streamlit](https://streamlit.io/)
        - [LangChain](https://python.langchain.com/)          
    ''')
    add_vertical_space(6)
    st.write("Made with 🤍 by TextNet")

def extract_key_value_pairs(text):
    # Enhanced regular expression to capture better key-value pairs
    pattern = r'([A-Za-z\s]+):\s*([^\n,]+)'  # Matches key: value
    matches = re.findall(pattern, text)
    
    # Create a DataFrame to store the key-value pairs
    if matches:
        data = pd.DataFrame(matches, columns=['Label', 'Value'])
    else:
        data = pd.DataFrame(columns=['Label', 'Value'])  # Empty DataFrame if no matches found
    return data

def main():
    st.header("Refining Dataset")
   
    load_dotenv()  # Load .env file to access environment variables
   
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        st.error("API Key not found. Please set the OpenAI API Key in the .env file.")
        return

    # Model selection with numbering from 1 to 4
    st.sidebar.subheader("Select ChatGPT Model")
    model_options = {
        "1 - gpt-3.5-turbo (Least Expensive)": {"name": "gpt-3.5-turbo", "input_cost": 0.0015, "output_cost": 0.002},
        "2 - gpt-3.5-turbo-16k": {"name": "gpt-3.5-turbo-16k", "input_cost": 0.003, "output_cost": 0.004},
        "3 - gpt-4": {"name": "gpt-4", "input_cost": 0.03, "output_cost": 0.06},
        "4 - gpt-4-32k (Most Expensive)": {"name": "gpt-4-32k", "input_cost": 0.06, "output_cost": 0.12}
    }
    model_display_names = list(model_options.keys())
    selected_model_display = st.sidebar.selectbox("Choose the model (1 = Least Expensive, 4 = Most Expensive):", model_display_names)
    selected_model_info = model_options[selected_model_display]
    selected_model = selected_model_info["name"]
    input_token_cost = selected_model_info["input_cost"] / 1000  # Convert to per token
    output_token_cost = selected_model_info["output_cost"] / 1000  # Convert to per token

    # Create placeholders for question and answer
    question_placeholder = st.empty()
    answer_placeholder = st.empty()
   
    # Upload the PDF file
    pdf = st.file_uploader("Upload your PDF", type="pdf")
   
    VectorStore = None  # Ensure VectorStore is initialized

    total_embedding_tokens = 0  # To track total tokens used in embeddings
    total_completion_tokens = 0  # To track total tokens used in completions

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        st.write(f"Uploaded: {pdf.name}")

        text = ""
       
        # Extract text from each page
        for page in pdf_reader.pages:
            text += page.extract_text() or ""

        # Add a feature to extract key-value pairs
        st.subheader("Extract Key-Value Pairs")
        if st.button("Extract Key-Value Pairs"):
            with st.spinner("Extracting key-value pairs..."):
                key_value_data = extract_key_value_pairs(text)
                
                # Display the extracted key-value pairs
                if not key_value_data.empty:
                    st.write(key_value_data)
                    
                    # Provide download options for CSV
                    csv_data = key_value_data.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download as CSV",
                        data=csv_data,
                        file_name="extracted_key_values.csv",
                        mime="text/csv"
                    )

                    # Provide download options for Excel using BytesIO
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        key_value_data.to_excel(writer, index=False, sheet_name="Key-Value Pairs")
                    processed_data = output.getvalue()

                    st.download_button(
                        label="Download as Excel",
                        data=processed_data,
                        file_name="extracted_key_values.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                else:
                    st.warning("No key-value pairs found in the document.")

        # Split the text into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=300,
            length_function=len
        )
        chunks = text_splitter.split_text(text)  
       
        store_name = pdf.name[:-4]
       
        # Define paths for FAISS index and metadata
        faiss_index_file = f"{store_name}_index.faiss"
        metadata_file = f"{store_name}_metadata.pkl"
       
        # Create the OpenAI Embeddings object
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)

        # Initialize the token encoder
        tokenizer = tiktoken.encoding_for_model('text-embedding-ada-002')

        if os.path.exists(faiss_index_file) and os.path.exists(metadata_file):
            # Load the FAISS index and metadata
            VectorStore = FAISS.load_local(faiss_index_file, embeddings, allow_dangerous_deserialization=True)
            with open(metadata_file, "rb") as f:
                chunks = pickle.load(f)  # Load additional metadata
            st.write("Embeddings Loaded from the Disk")
        else:
            # Calculate total tokens for embeddings
            for chunk in chunks:
                tokens = tokenizer.encode(chunk)
                total_embedding_tokens += len(tokens)
            # Compute the embeddings and save the FAISS index
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)  # Corrected parameter
           
            # Save the FAISS index and metadata
            VectorStore.save_local(faiss_index_file)
            with open(metadata_file, "wb") as f:
                pickle.dump(chunks, f)  # Save additional metadata
            st.write("Embeddings Computation Completed")

            # Calculate and print embedding cost
            embedding_cost = (total_embedding_tokens / 1000) * 0.0004  # $0.0004 per 1K tokens
            print(f"Total Embedding Tokens: {total_embedding_tokens}")
            print(f"Embedding Cost: ${embedding_cost:.6f}")

        # Pre-defined questions
        placeholder_question = "Select a question..."
        predefined_questions = [
            placeholder_question,
            "What are the main requirements outlined in the document?",
            "What are the project deliverables?",
            "What is the expected timeline for the project?",
            "What are the technical specifications?",
            "What are the budget constraints?",
            "Are there any risks identified?",
            "What are the key success criteria?",
            "Who are the stakeholders involved?",
            "What resources are required for the project?",
            "What are the legal or compliance considerations?"
        ]

        # Collapsible section for pre-written questions
        with st.expander("Pre-written Questions"):
            selected_question = st.radio("Select a question:", predefined_questions, index=0)

        # Input field for user queries
        query = st.text_input("Or ask your own question:")

        # Determine which query to use
        if query.strip() != "":
            query_to_use = query.strip()
        elif selected_question != placeholder_question:
            query_to_use = selected_question
        else:
            st.warning("Please enter a question in the text box or select one from the pre-written questions.")
            return

        # Check if VectorStore is available before using it
        if VectorStore and query_to_use:
            # Inform the user about the query being processed
            st.write(f"You asked: {query_to_use}")
            
            # Retrieve the most relevant documents
            docs = VectorStore.similarity_search(query=query_to_use, k=3)
           
            # Load QA chain with ChatOpenAI using selected model
            llm = ChatOpenAI(openai_api_key=api_key, model=selected_model)
            qa_chain = load_qa_chain(llm, chain_type="stuff")

            # Prepare the prompt and calculate tokens
            prompt_tokens = 0
            for doc in docs:
                prompt_tokens += len(tokenizer.encode(doc.page_content))
            prompt_tokens += len(tokenizer.encode(query_to_use))

            # Run the QA chain with the retrieved documents and the user's query
            answer = qa_chain.run(input_documents=docs, question=query_to_use)

            # Calculate tokens in the answer
            completion_tokens = len(tokenizer.encode(answer))

            total_completion_tokens = prompt_tokens + completion_tokens

            # Calculate and print completion cost
            completion_cost = (prompt_tokens * input_token_cost) + (completion_tokens * output_token_cost)

            print(f"Prompt Tokens: {prompt_tokens}")
            print(f"Completion Tokens: {completion_tokens}")
            print(f"Total Completion Tokens: {total_completion_tokens}")
            print(f"Completion Cost: ${completion_cost:.6f}")
            st.markdown("Total Computation Cost :")
            st.write("Total Completion Tokens: ", total_completion_tokens)
            st.write(f"Completion Cost: ${completion_cost:.6f}")

            total_cost = completion_cost
            if total_embedding_tokens > 0:
                embedding_cost = (total_embedding_tokens / 1000) * 0.0004
                total_cost += embedding_cost

            print(f"Total Estimated Cost: ${total_cost:.6f}")
           
            # Display the question and answer using placeholders
            question_placeholder.markdown(f"**Question:** {query_to_use}")
            answer_placeholder.markdown(f"**Answer:** {answer}")
    else:
        st.success("Upload your PDF to analyze it")

if __name__ == "__main__":
    main()
