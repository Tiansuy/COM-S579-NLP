from executor import MilvusExecutor
from executor import PipelineExecutor

import yaml
from easydict import EasyDict
import argparse

def read_yaml_config(file_path):
    with open(file_path, "r") as file:
        config_data = yaml.safe_load(file)
    return EasyDict(config_data)

class CommandLine():
    def __init__(self, config_path):
        self._mode = None
        self._executor = None
        self.config_path = config_path

    def show_start_info(self):
        with open('./start_info.txt') as fw:
            print(fw.read())

    def run(self):
        self.show_start_info()
        while True:
            conf = read_yaml_config(self.config_path)
            print('(rag) Select [milvus|pipeline]')
            # mode = input('(rag) ')
            mode = 'milvus'
            
            if mode == 'milvus':
                self._executor = MilvusExecutor(conf) 
                print('(rag) milvus has been selected')
                print('  1. Use `build [path]` to build corpus')
                print('  2. Use `ask` for existing index, `-d` ot enter debug mod')
                print('  3. Use`remove [file]` to delete existing index' )
                self._mode = 'milvus'
                break
            elif mode == 'pipeline':
                self._executor = PipelineExecutor(conf)
                print('(rag) pipeline has been selected')
                print('  1. Use `build https://raw.githubusercontent.com/wxywb/history_rag/master/[path]` to build corpus')
                print('  2. Use `ask` for existing index, `-d` ot enter debug mod')
                print('  3. Use`remove [file]` to delete existing index' )
                self._mode = 'pipeline'
                break
            elif mode == 'quit':
                self._exit()
                break
            else:
                print(f'(rag) {mode} is not known scheme，select one of [milvus|pipeline] or use `quit` to exit')
        assert self._mode != None
        while True:
            command_text = input("(rag) ")
            self.parse_input(command_text)

    def parse_input(self, text):
        commands = text.split(' ')
        if commands[0] == 'build':
            if len(commands) == 3:
                if commands[1] == '-overwrite':  
                    print(commands)
                    self.build_index(path=commands[2], overwrite=True)
                else:
                    print('(rag) build only support `-overwrite` parameter')
            elif len(commands) == 2:
                self.build_index(path=commands[1], overwrite=False)
        elif commands[0] == 'ask':
            if len(commands) == 2:
                if commands[1] == '-d':
                    self._executor.set_debug(True)
                else: 
                    print('(rag) ask only support `-d` parameter ')
            else:
                self._executor.set_debug(False)
            self.question_answer()
        elif commands[0] == 'remove':
            if len(commands) != 2:
                print('(rag) remove only accept 1 parameter')
            self._executor.delete_file(commands[1])
            
        elif 'quit' in commands[0]:
            self._exit()
        else: 
            print('(rag) Select one of [build|ask|remove|quit] to operate, please try again.')
            
    def query(self, question):
        ans = self._executor.query(question)
        print(ans)
        print('+---------------------------------------------------------------------------------------------------------------------+')
        print('\n')

    def build_index(self, path, overwrite):
        self._executor.build_index(path, overwrite)
        print('(rag) Index building completed')

    def remove(self, filename):
        self._executor.delete_file(filename)
        
    def question_answer(self):
        self._executor.build_query_engine()
        while True: 
            question = input("(rag) Ask: ")
            if question == 'quit':
                print('(rag) Exit Q&A')
                break
            elif question == "":
                continue
            else:
                pass
            self.query(question)

    def _exit(self):
        exit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='Path to the configuration file', default='cfgs/config.yaml')
    args = parser.parse_args()

    cli = CommandLine(args.cfg)
    cli.run()