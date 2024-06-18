from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.chains.summarize import load_summarize_chain
from transformers import T5Tokenizer,T5ForConditionalGeneration
from transformers import pipeline
import torch 
import base64 
import streamlit as st 




# model and tokenizer 

checkpoint = "LaMini-Flan-T5-248M"
base_model = T5ForConditionalGeneration.from_pretrained(checkpoint,device_map='auto',torch_dtype=torch.float32)
#file loader and preprocessing 
def file_processing(file):
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=50)
    texts = text_splitter.split_documents(pages)
    final_text = "" 
    for text in texts:
        print(text)
        final_text = final_text + text.page_content
    return final_text



#LM pipeline 
def llm_pipeline(filepath):
    pipe_sum = pipeline(
        "summarization",
        model=base_model,
        tokenizer=tokenizer,
        device=0,
        max_length=1000,
        min_length=50,
        do_sample=False,
        top_k=50,
        top_p=0.95,
        num_return_sequences=1,
    )

    input_text = file_processing(filepath)
    result = pipe_sum(input_text)
    return result[0]['summary_text']


#streamlit code 


@st.cache_data 
def displayPdf(file):
    with open(file,"rb") as f :
        base_pdf = base64.b64decode(f.read()).decode('utf-8')
    
    #embedding PDF in HTML 

    pdf_display = F'<iframe src="data:application/pdf;base64,{base64}" width="100%" height="600" type="application/pdf"></iframe>'

    st.markdown(pdf_display,unsafe_allow_html = True)

st.set_page_config(layout='wide',page_title="Summarization App")

def main():
    st.title("Document Summarization App using Language Model")

    uploaded_file = st.file_uploader("Upload the PDF",type=["pdf"])

    if uploaded_file is not None:
        if st.button("Summarize"):
            col1,col2 = st.columns(2)
            filepath = "data/"+uploaded_file.name
            with open(filepath,'wb') as temp_file:
                temp_file.write(uploaded_file.read())
            with col1:
                st.info("Upload PDF file")
                pdf_viewer = displayPdf(filepath)

            
            with col2:
                st.info("Summarization is below")
                summary = llm_pipeline(filepath)
                st.success(summary)

if __name__ == 'main':
    main()