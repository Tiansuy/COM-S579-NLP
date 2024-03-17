import argparse
import logging
import sys
import re
import os
import argparse

import requests
from pathlib import Path
from urllib.parse import urlparse

from llama_index import ServiceContext, StorageContext
from llama_index import set_global_service_context
from llama_index import VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index.llms import OpenAI
from llama_index.readers.file.flat_reader import FlatReader
from llama_index.readers import PDFReader
from llama_index.vector_stores import MilvusVectorStore
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.node_parser.text import SentenceWindowNodeParser

from llama_index.prompts import ChatPromptTemplate, ChatMessage, MessageRole, PromptTemplate
from llama_index.postprocessor import MetadataReplacementPostProcessor
from llama_index.postprocessor import SentenceTransformerRerank
#from llama_index.indices import ZillizCloudPipelineIndex
from llama_index.indices.query.schema import QueryBundle
from llama_index.schema import BaseNode, ImageNode, MetadataMode

# from src.zilliz.base import ZillizCloudPipelineIndex
from src.history_sentence_window import HistorySentenceWindowNodeParser
# from src.llms.QwenLLM import QwenUnofficial
# from src.llms.GeminiLLM import Gemini

from pymilvus import MilvusClient

QA_PROMPT_TMPL_STR = (
    "Please carefully read the relevant content and provide answers based on the materials. Each piece of information should be labeled in the form of 'From: <Title>' \
        (if you answer, please cite the original text clearly and accurately, give your answer first, then attach the corresponding original text, and use the Book Title to label the original text). If you find that the information cannot be answered, you will answer 'I don't know' \n"
    "The relevant information searched is as follows.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Question: {query_str}\n"
    "Answer: "
)

QA_SYSTEM_PROMPT = "You are a rigorous knowledge question answering agent. You will carefully read the materials and provide accurate answers. Your answers will be very accurate because after answering, \
    you will use the evidence provided in <Title> [] to support your answer. You will also explain at the beginning whether the original text has the necessary knowledge to answer"

REFINE_PROMPT_TMPL_STR = ( 
    "You are a knowledge answering and correction robot, and you strictly work in the following way"
    "1. Only make corrections when the original answer is unknown, otherwise output the content of the original answer\n"
    "2. When making revisions, in order to demonstrate your accuracy and objectivity, you really enjoy using <Title> [] to present the original text.\n"
    "3. If you feel confused, answer with the content of the original answer."
    "New knowledge: {context_msg}\n"
    "Question: {query_str}\n"
    "Original Answer: {existing_answer}\n"
    "New Answer: "
)

def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def is_github_folder_url(url):
    return url.startswith('https://raw.githubusercontent.com/') and '.' not in os.path.basename(url)


def get_branch_head_sha(owner, repo, branch):
    url = f"https://api.github.com/repos/{owner}/{repo}/git/ref/heads/{branch}"
    response = requests.get(url)
    data = response.json()
    sha = data['object']['sha']
    return sha

def get_github_repo_contents(repo_url):
    # repo_url example: https://raw.githubusercontent.com/wxywb/history_rag/master/data/history_24/
    repo_owner = repo_url.split('/')[3]
    repo_name = repo_url.split('/')[4]
    branch = repo_url.split('/')[5]
    folder_path = '/'.join(repo_url.split('/')[6:])
    sha = get_branch_head_sha(repo_owner, repo_name, branch)
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/git/trees/{sha}?recursive=1"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()

            raw_urls = []
            for file in data['tree']:
                if file['path'].startswith(folder_path) and file['path'].endswith('.txt'):
                    raw_url = f"https://raw.githubusercontent.com/{repo_owner}/{repo_name}/{branch}/{file['path']}"
                    raw_urls.append(raw_url)
            return raw_urls
        else:
            print(f"Failed to fetch contents. Status code: {response.status_code}")
    except Exception as e:
        print(f"Failed to fetch contents. Error: {str(e)}")
    return []

class Executor:
    def __init__(self, model):
        pass

    def build_index(self, path, overwrite):
        pass

    def build_query_engine(self):
        pass
     
    def delete_file(self, path):
        pass
    
    def query(self, question):
        pass
 

class MilvusExecutor(Executor):
    def __init__(self, config):
        self.index = None
        self.query_engine = None
        self.config = config
        self.node_parser = HistorySentenceWindowNodeParser.from_defaults(
            sentence_splitter=lambda text: re.findall("[^,.;。？！]+[,.;。？！]?", text),
            # sentence_splitter=lambda text: re.findall("[^\s,.;。？！]+[,.;。？！]?", text),
            window_size=config.milvus.window_size,
            window_metadata_key="window",
            original_text_metadata_key="original_text",)

        embed_model = HuggingFaceEmbedding(model_name=config.embedding.name)

        # 使用Qwen 通义千问模型
        if config.llm.name.find("qwen") != -1:
            llm = QwenUnofficial(temperature=config.llm.temperature, model=config.llm.name, max_tokens=2048)
        elif config.llm.name.find("gemini") != -1:
            llm = Gemini(temperature=config.llm.temperature, model_name=config.llm.name, max_tokens=2048)
        else:
            api_base = None
            if 'api_base' in config.llm:
                api_base = config.llm.api_base
            llm = OpenAI(api_base = api_base, temperature=config.llm.temperature, model=config.llm.name, max_tokens=2048)

        service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
        set_global_service_context(service_context)
        rerank_k = config.milvus.rerank_topk
        self.rerank_postprocessor = SentenceTransformerRerank(
            model=config.rerank.name, top_n=rerank_k)
        self._milvus_client = None
        self._debug = False
        
    def set_debug(self, mode):
        self._debug = mode

    def build_index(self, path, overwrite):
        config = self.config
        vector_store = MilvusVectorStore(
            uri = f"http://{config.milvus.host}:{config.milvus.port}",
            collection_name = config.milvus.collection_name,
            overwrite=overwrite,
            dim=config.embedding.dim)
        self._milvus_client = vector_store.milvusclient
        if os.path.exists(path) is False:
            print(f'(rag) File not find{path}')
            return
        elif path.endswith('.pdf'):
            documents = PDFReader().load_data(Path(path))
            for i in range(1,len(documents)):
                documents[0].text+=' '+documents[i].text
            documents[0].text = documents[0].text.replace('\n','')
            def add_newline_after_nth_period(text, n):
                # sentences = text.split('。')
                sentences = re.split(r'[。\.]', text)
                if sentences[-1] == '':
                    sentences.pop()
                for i in range(n-1, len(sentences), n):
                    sentences[i] += '\n'
                return '.'.join(sentences)
            documents[0].text = add_newline_after_nth_period(documents[0].text, 5)
            documents = [documents[0]]
            documents[0].metadata = {
                "filename":documents[0].metadata['file_name'].split('\\')[-1],
                "extension":'.pdf'
            }
            documents[0].metadata['file_name'] = documents[0].metadata['filename']
        elif path.endswith('.txt'):
            documents = FlatReader().load_data(Path(path))
            documents[0].metadata['file_name'] = documents[0].metadata['filename'] 
        elif os.path.isfile(path):           
            print('(rag) Only support txt\pdf file')
        elif os.path.isdir(path):
            if os.path.exists(path) is False:
                print(f'(rag) Path not find{path}')
                return
            else:
                documents = SimpleDirectoryReader(path).load_data()
        else:
            return

        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        nodes = self.node_parser.get_nodes_from_documents(documents)
        self.index = VectorStoreIndex(nodes, storage_context=storage_context, show_progress=True)

    def _get_index(self):
        config = self.config
        vector_store = MilvusVectorStore(
            uri = f"http://{config.milvus.host}:{config.milvus.port}",
            collection_name = config.milvus.collection_name,
            dim=config.embedding.dim)
        self.index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        self._milvus_client = vector_store.milvusclient

    def build_query_engine(self):
        config = self.config
        if self.index is None:
            self._get_index()
        self.query_engine = self.index.as_query_engine(node_postprocessors=[
            self.rerank_postprocessor,
            MetadataReplacementPostProcessor(target_metadata_key="window")
        ])
        self.query_engine._retriever.similarity_top_k=config.milvus.retrieve_topk

        message_templates = [
            ChatMessage(content=QA_SYSTEM_PROMPT, role=MessageRole.SYSTEM),
            ChatMessage(
                content=QA_PROMPT_TMPL_STR,
                role=MessageRole.USER,
            ),
        ]
        chat_template = ChatPromptTemplate(message_templates=message_templates)
        self.query_engine.update_prompts(
            {"response_synthesizer:text_qa_template": chat_template}
        )
        self.query_engine._response_synthesizer._refine_template.conditionals[0][1].message_templates[0].content = REFINE_PROMPT_TMPL_STR

    def delete_file(self, path):
        config = self.config
        if self._milvus_client is None:
            self._get_index()
        num_entities_prev = self._milvus_client.query(collection_name='history_rag',filter="",output_fields=["count(*)"])[0]["count(*)"]
        res = self._milvus_client.delete(collection_name=config.milvus.collection_name, filter=f"file_name=='{path}'")
        num_entities = self._milvus_client.query(collection_name='history_rag',filter="",output_fields=["count(*)"])[0]["count(*)"]
        print(f'(rag) {num_entities} data existing，delete {num_entities_prev - num_entities} data')
    
    def query(self, question):
        if self.index is None:
            self._get_index()
        if question.endswith('?') or question.endswith('？'):
            question = question[:-1]
        if self._debug is True:
            contexts = self.query_engine.retrieve(QueryBundle(question))
            for i, context in enumerate(contexts): 
                print(f'{question}', i)
                content = context.node.get_content(metadata_mode=MetadataMode.LLM)
                print(content)
            print('-------------------------------------------------------Reference---------------------------------------------------------')
        response = self.query_engine.query(question)
        return response

class PipelineExecutor(Executor):
    def __init__(self, config):
        self.ZILLIZ_CLUSTER_ID = os.getenv("ZILLIZ_CLUSTER_ID")
        self.ZILLIZ_TOKEN = os.getenv("ZILLIZ_TOKEN")
        self.ZILLIZ_PROJECT_ID = os.getenv("ZILLIZ_PROJECT_ID") 
        self.ZILLIZ_CLUSTER_ENDPOINT = f"https://{self.ZILLIZ_CLUSTER_ID}.api.gcp-us-west1.zillizcloud.com"
    
        self.config = config
        if len(self.ZILLIZ_CLUSTER_ID) == 0:
            print('ZILLIZ_CLUSTER_ID is None')
            exit()

        if len(self.ZILLIZ_TOKEN) == 0:
            print('ZILLIZ_TOKEN is None')
            exit()
        
        self.config = config
        self._debug = False

        # if config.llm.name.find("qwen") != -1:
        #     llm = QwenUnofficial(temperature=config.llm.temperature, model=config.llm.name, max_tokens=2048)
        # elif config.llm.name.find("gemini") != -1:
        #     llm = Gemini(model_name=config.llm.name, temperature=config.llm.temperature, max_tokens=2048)
        # else:
        #     api_base = None
        #     if 'api_base' in config.llm:
        #         api_base = config.llm.api_base
        #     llm = OpenAI(api_base = api_base, temperature=config.llm.temperature, model=config.llm.name, max_tokens=2048)
        api_base = None
        if 'api_base' in config.llm:
            api_base = config.llm.api_base
        llm = OpenAI(api_base = api_base, temperature=config.llm.temperature, model=config.llm.name, max_tokens=2048)

        service_context = ServiceContext.from_defaults(llm=llm, embed_model=None)
        self.service_context = service_context
        set_global_service_context(service_context)
        self._initialize_pipeline(service_context)

        #rerank_k = config.rerankl
        #self.rerank_postprocessor = SentenceTransformerRerank(
        #    model="BAAI/bge-reranker-large", top_n=rerank_k)

    def set_debug(self, mode):
        self._debug = mode

    def _initialize_pipeline(self, service_context: ServiceContext):
        config = self.config
        try:
            # self.index = ZillizCloudPipelineIndex(
            #     project_id = self.ZILLIZ_PROJECT_ID,
            #     cluster_id=self.ZILLIZ_CLUSTER_ID,
            #     token=self.ZILLIZ_TOKEN,
            #     collection_name=config.pipeline.collection_name,
            #     service_context=service_context,
            #  )
            if len(self._list_pipeline_ids()) == 0:
                self.index.create_pipelines(
                    metadata_schema={"digest_from":"VarChar"}, chunk_size=self.config.pipeline.chunk_size
                )
        except Exception as e:
            print('(rag) zilliz pipeline connection exception', str(e))
            exit()
        try:
            self._milvus_client = MilvusClient(
                uri=self.ZILLIZ_CLUSTER_ENDPOINT, 
                token=self.ZILLIZ_TOKEN 
            )
        except Exception as e:
            print('(rag) zilliz cloud connection exception', str(e))

    def build_index(self, path, overwrite):
        config = self.config
        if not is_valid_url(path) or 'github' not in path:
            print('(rag) It is an illegal url，please try `https://raw.githubusercontent.com/wxywb/history_rag/master/{path}` again')
            return
        if overwrite == True:
            self._milvus_client.drop_collection(config.pipeline.collection_name)
            pipeline_ids = self._list_pipeline_ids()
            self._delete_pipeline_ids(pipeline_ids)

            self._initialize_pipeline(self.service_context)

        if is_github_folder_url(path):
            urls = get_github_repo_contents(path)
            for url in urls:
                print(f'(rag) Building index {url}')
                self.build_index(url, False)  # already deleted original collection
        elif path.endswith('.txt'):
            self.index.insert_doc_url(
                url=path,
                metadata={"digest_from": HistorySentenceWindowNodeParser.book_name(os.path.basename(path))},
            )
        else:
            print('(rag) Only support txt or folders end with on GitHub.')

    def build_query_engine(self):
        config = self.config
        self.query_engine = self.index.as_query_engine(
          search_top_k=config.pipeline.retrieve_topk)
        message_templates = [
            ChatMessage(content=QA_SYSTEM_PROMPT, role=MessageRole.SYSTEM),
            ChatMessage(
                content=QA_PROMPT_TMPL_STR,
                role=MessageRole.USER,
            ),
        ]
        chat_template = ChatPromptTemplate(message_templates=message_templates)
        self.query_engine.update_prompts(
            {"response_synthesizer:text_qa_template": chat_template}
        )
        self.query_engine._response_synthesizer._refine_template.conditionals[0][1].message_templates[0].content = REFINE_PROMPT_TMPL_STR


    def delete_file(self, path):
        config = self.config
        if self._milvus_client is None:
            self._get_index()
        num_entities_prev = self._milvus_client.query(collection_name='history_rag',filter="",output_fields=["count(*)"])[0]["count(*)"]
        res = self._milvus_client.delete(collection_name=config.milvus.collection_name, filter=f"doc_name=='{path}'")
        num_entities = self._milvus_client.query(collection_name='history_rag',filter="",output_fields=["count(*)"])[0]["count(*)"]
        print(f'(rag) {num_entities} data existing，delete {num_entities_prev - num_entities} data')

    def query(self, question):
        if self.index is None:
            self.get_index()
        if question.endswith("?") or question.endswith("？"):
            question = question[:-1]
        if self._debug is True:
            contexts = self.query_engine.retrieve(QueryBundle(question))
            for i, context in enumerate(contexts): 
                print(f'{question}', i)
                content = context.node.get_content(metadata_mode=MetadataMode.LLM)
                print(content)
            print('-------------------------------------------------------Reference---------------------------------------------------------')
        response = self.query_engine.query(question)
        return response

    def _list_pipeline_ids(self):
        url = f"https://controller.api.gcp-us-west1.zillizcloud.com/v1/pipelines?projectId={self.ZILLIZ_PROJECT_ID}"
        headers = {
            "Authorization": f"Bearer {self.ZILLIZ_TOKEN}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        collection_name = self.config.milvus.collection_name
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise RuntimeError(response.text)
        response_dict = response.json()
        if response_dict["code"] != 200:
            raise RuntimeError(response_dict)
        pipeline_ids = []
        for pipeline in response_dict['data']: 
            if collection_name in  pipeline['name']:
                pipeline_ids.append(pipeline['pipelineId'])
            
        return pipeline_ids

    def _delete_pipeline_ids(self, pipeline_ids):
        for pipeline_id in pipeline_ids:
            url = f"https://controller.api.gcp-us-west1.zillizcloud.com/v1/pipelines/{pipeline_id}/"
            headers = {
                "Authorization": f"Bearer {self.ZILLIZ_TOKEN}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            }

            response = requests.delete(url, headers=headers)
            if response.status_code != 200:
                raise RuntimeError(response.text)