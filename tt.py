# from openai import OpenAI
# client = OpenAI()

# completion = client.chat.completions.create(
#   model="gpt-3.5-turbo",
#   messages=[
#     {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
#     {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
#   ]
# )

# print(completion.choices[0].message)

from llama_index.readers.file.flat_reader import FlatReader
from llama_index.readers import PDFReader
# from llama_index.readers.file.docs_reader
from pathlib import Path
import os
# import pdfminer.high_level


# def build_index(self, path, overwrite):
#     config = self.config
#     vector_store = MilvusVectorStore(
#         uri=f"http://{config.milvus.host}:{config.milvus.port}",
#         collection_name=config.milvus.collection_name,
#         overwrite=overwrite,
#         dim=config.embedding.dim)
#     self._milvus_client = vector_store.milvusclient
    
#     documents = []

#     if path.endswith('.pdf'):
#         if not os.path.exists(path):
#             print(f'(rag) 没有找到文件 {path}')
#             return
#         text = extract_text_from_pdf(path)
#         # Assuming FlatReader can handle string input for consistency,
#         # or you would need to adapt this step to fit your data handling.
#         documents = FlatReader().load_data_from_string(text)
#         documents[0].metadata['file_name'] = path


# def extract_text_from_pdf(pdf_path):
#     """Extract text from a PDF file."""
#     return pdfminer.high_level.extract_text(pdf_path)

# text = extract_text_from_pdf("data/menu.pdf")
# print(text)
documents = PDFReader().load_data(Path('data/menu.pdf'))
for i in range(1,len(documents)):
    documents[0].text+='。'+documents[i].text
documents = [documents[0]]
documents[0].metadata = {
    "filename":documents[0].metadata['file_name'].split('\\')[-1],
    "extension":'.pdf'
}
documents[0].metadata['file_name'] = documents[0].metadata['filename']
# documents[0].text = ''

documents2 = FlatReader().load_data(Path('data/test.txt'))
documents2[0].metadata['file_name'] = documents[0].metadata['filename'] 
documents2[0].text = ''

print(documents)
print(len(documents))
print('\n\n\n\n')
print(documents2)
print(len(documents2))