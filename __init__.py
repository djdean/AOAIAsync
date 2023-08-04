import logging
import json
import azure.functions as func
from azure.storage.blob import BlobServiceClient
from langchain.llms import AzureOpenAI
import openai
from langchain import LLMChain, PromptTemplate
import asyncio
import time

def main(req: func.HttpRequest) -> func.HttpResponse:

    openai.api_key = "<AOAI API KEY>"
    openai.api_base =  "<AOAI_ebdpoint>"
    openai.api_type = 'azure'
    openai.api_version = '2022-12-01' # this may change in the future
    model_deployment_name = "speechsummary"
    model_name = "<AOAI Model Name>"
    logging.info('Python HTTP trigger function processed a request.')
    connection_string = "Azure Storage Connection String"
    blob_service_client = get_blob_service_client_connection_string(connection_string)
    req_body = req.get_json()
    container_name = req_body.get('container_name')
    transcription_metadata_file = req_body.get('transcription_metadata_file')
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=transcription_metadata_file)
    blob_data = blob_client.download_blob()
    data_json = json.loads(blob_data.readall().decode('ascii'))
    file_list = data_json['file_names']
    prompt_input = req_body.get('prompt')
    batch_size = req_body.get('batch_size')
    if batch_size is None:
        batch_size = 120
    else:
        batch_size = int(batch_size)
    #initializing chat instance from langchain
    llm = AzureOpenAI(
        openai_api_base=openai.api_base,
        openai_api_version=openai.api_version,
        deployment_name=model_deployment_name,
        openai_api_key=openai.api_key,
        openai_api_type=openai.api_type,
        model_name=model_name,
        max_tokens=1000, 
        temperature=0
        )
    prompt_template = "Based on the following call transcript:\ \n\n--------------Begin Transcript-------------\n{content}\n--------------End Transcript-------------\n\n{prompt}"
    prompt = PromptTemplate(
        input_variables=["content","prompt"], template=prompt_template
    )
    
    asyncio.run(process_files_async(llm, prompt, prompt_input, blob_service_client, container_name, 
                                    file_list, transcription_metadata_file, model_deployment_name, batch_size=batch_size))


    return func.HttpResponse(
        "Completed run.\n",
        status_code=200
    )



async def process_files_async(llm, prompt, prompt_input, blob_service_client, container_name, file_list, transcription_metadata_file, model_deployment_name, batch_size):
    tasks = []
    processed = 0
    for current_file in file_list:
        try:
            if check_process_file(transcription_metadata_file, current_file):
                processed += 1
                logging.info("Processing transcript: " + current_file)
                blob_client = blob_service_client.get_blob_client(container=container_name, blob=current_file)
                blob_data = blob_client.download_blob()
                data_json = json.loads(blob_data.readall())
                transcript_content = data_json['combinedRecognizedPhrases'][0]['lexical']
                chain = LLMChain(llm=llm, prompt=prompt)
                tasks.append(async_run_aoai(chain, prompt_input, transcript_content, current_file, model_deployment_name, blob_service_client, container_name, blob_client))
                #AOAI has a limit of 120 calls per minute
                if processed == batch_size:
                    time.sleep(60)
                    await asyncio.gather(*tasks)
                    tasks = []
                    processed = 0
        except:
            logging.info("Error processing transcript: " + current_file)
            blob_client.close()
        #Process any remaining files
    await asyncio.gather(*tasks)

async def async_run_aoai(chain,prompt_input, transcript_content,current_file, model_deployment_name, blob_service_client, container_name, blob_client):
    try:
        resp = await chain.arun({
                    "prompt": prompt_input,
                    "content": transcript_content
                }) 
        resp_parsed = resp.replace("\n","").replace("\'","\"")
        output_json = json.loads(resp_parsed)
        output_json['current_file_name'] = current_file
        output_json['model_deployment_name'] = model_deployment_name
        file_name_split = current_file.split(".")
        output_file_name = file_name_split[0] + "_with_openAI.json"
        output_json_str = json.dumps(output_json)
        blob_client.close()
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=output_file_name)
        blob_client.upload_blob(output_json_str, overwrite=True)
    except:
        logging.info("Error processing transcript: " + current_file)
        blob_client.close()
def check_process_file(metadata_filename, current_filename):
    if metadata_filename == current_filename:
        return False
    elif not ".json" in current_filename:
        return False
    elif "_with_openAI" in current_filename:
        return False
    return True
def get_blob_service_client_connection_string(connection_string):
    
    # Create the BlobServiceClient object
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)

    return blob_service_client