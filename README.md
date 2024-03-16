# COM-S579-NLP
COM-S579-NLP-course-project-RAG

# requirements
- llama-index==0.9.39
- pymilvus
- easydict
- dashscope

- torch
- transformers
- pyyaml
- gradio

- pypdf

# how to use(command line tool)
1. clone repo
2. install docker
3. install milvus with docker
```
cd db
sudo docker compose up -d
```
4. install python dependencies(recommend conda env)
```
pip install -r requirements.txt
```
5. set up openai api key to environment variable
```
setx OPENAI_API_KEY "your-api-key-here"
```

6. build index using your file
format:"build your_file_path"
```
python cli.py
(rag) build ./data/menu.pdf
```

7. other features
    - remove index: "remove filename"
    ```
    (rag) build menu.pdf
    ```
    - query: "ask"
    ```
    (rag) ask
    (rag) 问题:年年有余怎么做？
    ```
# how to use(GUI)
- to be finished

