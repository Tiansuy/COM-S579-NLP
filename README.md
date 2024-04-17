# COM-S579-NLP
COM-S579-NLP-course-project-RAG

Demo video link: https://youtu.be/uP9FPIQGEw4

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
(rag) build ./data/bitcoin.pdf
```

7. other features
    - remove index: "remove filename"
    ```
    (rag) remove bitcoin.pdf
    ```
    - query: "ask"
    ```
    (rag) ask
    (rag) Question: what is bitcoin？
    ```
# how to use(GUI)

GUI demo video: https://github.com/Tiansuy/COM-S579-NLP

```
python gradioui.py
```

