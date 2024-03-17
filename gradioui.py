from executor import MilvusExecutor
from executor import PipelineExecutor

import gradio as gr

from cli import CommandLine, read_yaml_config  # 导入 CommandLine 类

resolutions = ["milvus", "pipeline"]

build_tasks = ["Build index", "Delete index"]
query_tasks = ["Ask", "Ask + Return to retrieve content"]


class GradioCommandLine(CommandLine):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.config_path = cfg

    def index(self, task, path, overwrite):
        if task == "Build index":
            self._executor.build_index(path, overwrite)
            return "Index building completed"
        elif task == "Delete index":
            self._executor.delete_file(path)
            return "Index deletion completed"

    def query(self, task, question):
        if task == "Ask":
            return self._executor.query(question)
        elif task == "Ask + Return to retrieve content":
            self._executor.set_debug(True)
            return self._executor.query(question)


def initialize_cli(cfg_path, resolution):
    global cli_instance
    cli_instance = GradioCommandLine(cfg_path)
    conf = read_yaml_config(cli_instance.config_path)
    if resolution == "milvus":
        cli_instance._executor = MilvusExecutor(conf)
        cli_instance._mode = "milvus"
    else:
        cli_instance._executor = PipelineExecutor(conf)
        cli_instance._mode = "pipeline"
    cli_instance._executor.build_query_engine()
    return "CLI nitialization completed"


with gr.Blocks() as demo:
    # 初始化
    gr.Interface(fn=initialize_cli,
                 inputs=[gr.Textbox(
                     lines=1, value="cfgs/config.yaml"),
                     gr.Dropdown(resolutions, label="Index categories", value="milvus")],
                 outputs="text",
                 submit_btn="Initialization", clear_btn="Clear")
    # 构建索引
    gr.Interface(fn=lambda command, argument, overwrite: cli_instance.index(command, argument, overwrite),
                 inputs=[gr.Dropdown(choices=build_tasks, label="Choose commend", value="Build index"),
                         gr.Textbox(label="Path"), gr.Checkbox(label="Recover index")], outputs="text",
                 submit_btn="Submit", clear_btn="Clear")

    # 提问
    gr.Interface(fn=lambda command, argument: cli_instance.query(command, argument),
                 inputs=[gr.Dropdown(choices=query_tasks, label="Choose commend", value="Ask"),
                         gr.Textbox(label="Question")], outputs="text",
                 submit_btn="Submit", clear_btn="Clear")
    with open("docs/web_ui.md", "r", encoding="utf-8") as f:
        article = f.read()
    gr.Markdown(article)

if __name__ == '__main__':
    # 启动 Gradio 界面
    demo.launch()
