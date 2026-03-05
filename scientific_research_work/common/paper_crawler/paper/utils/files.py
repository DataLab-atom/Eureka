import re

class Files:
    def __init__(self):
        pass

    @staticmethod
    def sanitize_filename(filename):
        """移除或替换文件名中的非法字符"""
        filename = filename.replace('{', '').replace('}', '').replace('\n', ' ')
        return re.sub(r'[\\/*?:"<>|]', "", filename)