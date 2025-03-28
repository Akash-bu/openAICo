from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import concurrent
import PyPDF2
import os 
import pandas as pd
import base64

client = OpenAI(api_key = os.getenv('OPENAI_API_KEY'))
pdf_dir = 'openai_blog_pdfs'
pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith('.pdf')]

def upload_single_pdf(file_path: str, vector_store_id: str):
    file_name = os.path.basename(file_path)
    try:
        file_response = client.files.create(file=open(file_path, 'rb'), purpose = "assistants")
        attach_response = client.vector_stores.files.create(
            vector_store_id = vector_store_id,
            file_id = file_response.id
        )
        return {"file": file_name, "status": "success"}
    except Exception as e:
        print(f"Error with {file_name}: {str(e)}")
        return {"file": file_name, "status": "failed", "error":str(e)}

def upload_pdf_files_to_vector_store(vector_store_id: str):
    pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir)]
    stats = {"total_files": len(pdf_files), "successful_uploads": 0, "failed_uploads": 0, "errors": []}

    print(f"{len(pdf_files)} PDF files to process. Uploading in parallel...")

    with concurrent.futures.ThreadPoolExecutor(max_workers = 10) as executor:
        futures = {executor.submit(upload_single_pdf, file_path, vector_store_id): file_path for file_path in pdf_files} 
        for future in tqdm(concurrent.futures.as_completed(futures), total = len(pdf_files)):
            
            result = future.result()
            if result["status"] == "success":
                stats["successful_uploads"] += 1
            else:
                stats["failed_uploads"] += 1
                stats["errors"].append(result)

    return stats

def create_vector_store(store_name: str) -> dict:
    try:
        vector_store = client.vector_stores.create(name = store_name)
        details = {
            "id": vector_store.id,
            "name": vector_store.name,
            "created_at": vector_store.created_at,
            "file_count": vector_store.file_counts.completed
        }  

        print("vector store created successfully: ", details)
        return details
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return {}

store_name = "openai_blog_store"
vector_store_details = create_vector_store(store_name)
upload_pdf_files_to_vector_store(vector_store_details["id"])

#query = "What are the New risks from native image generation?"


# search_results = client.vector_stores.search(
#     vector_store_id = vector_store_details['id'],
#     query = query
# )

# for result in search_results.data:
#     print(str(len(result.content[0].text)) + ' of character of content from ' + result.filename + ' with a relevant score of ' + str(result.score))

#Integrating search results with LLM in a single API call

query = " What is Catching Systemic Reward Hacking?"

response = client.responses.create(
    input = query,
    model = "gpt-4o-mini",
    tools=[{
        "type": "file_search",
        "vector_store_ids": [vector_store_details['id']]
    }]
    
)

# print(response)

# # Extract annotations from the response
# annotations = response.output[1].content[0].annotations
    
# # Get top-k retrieved filenames
# retrieved_files = set([result.filename for result in annotations])

# print(f'Files used: {retrieved_files}')
# print('Response:')
# print(response.output[1].content[0].text) # 0 being the filesearch call

# Assuming 'response' is the response object you received
response_output_message = response.output[2]  # Access the ResponseOutputMessage
annotations = response_output_message.content[0].annotations  # Access annotations in the content

# Print the annotations to check the results
for annotation in annotations:
    print(f"File: {annotation.filename}, Index: {annotation.index}")

text = response_output_message.content[0].text

print(text)
