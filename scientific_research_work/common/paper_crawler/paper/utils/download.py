import os
import requests

from llm.env import get_env
from .files import Files
# import logging

USER_AGENT = get_env("USER_AGENT") or "" # 设置User-Agent

DOWNLOAD_TIMEOUT = 30  # 下载超时时间 (秒)

PDF_MD_TOKEN = get_env("PDF_MD_TOKEN") or ""

class Download:
    def __init__(self):
        pass

    @staticmethod
    def download_pdf(url, filename, folder):
        """下载指定URL的PDF文件"""
        if not url:
            return False
        headers = {'User-Agent': USER_AGENT}
        try:
            print(f"  [INFO] 尝试从 {url} 下载...")
            response = requests.get(url, headers=headers, timeout=DOWNLOAD_TIMEOUT, stream=True)
            response.raise_for_status() # 如果状态码不是200，则引发HTTPError

            # 检查内容类型是否为PDF
            content_type = response.headers.get('Content-Type', '').lower()
            if 'application/pdf' not in content_type:
                print(f"  [WARN] 链接 {url} 返回的不是PDF (Content-Type: {content_type})")
                return False

            filepath = os.path.join(folder, Files.sanitize_filename(f"{filename}.pdf"))
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"  [SUCCESS] 文件已保存到: {filepath}")
            return True
        except requests.exceptions.HTTPError as e:
            print(f"  [FAIL] HTTP错误 {e.response.status_code} 下载 {url}: {e}")
        except requests.exceptions.ConnectionError as e:
            print(f"  [FAIL] 连接错误下载 {url}: {e}")
        except requests.exceptions.Timeout:
            print(f"  [FAIL] 下载超时 {url}")
        except Exception as e:
            print(f"  [FAIL] 下载时发生未知错误 {url}: {e}")
        return False
    
    @staticmethod
    def download_zip(url, filename, folder):
        """上传URL并下载压缩包"""
        if not url:
            return False
        
        # 暂时硬编码
        token = PDF_MD_TOKEN
        base_url = "https://mineru.net/api/v4/extract/task"
        header = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
        data = {
            "url": url,
            'language': 'en'
                 }

        res = requests.post(base_url,headers=header,json=data)
        if res.status_code!=200:
            print(f"[FAIL] {url}上传失败！！！！！")
            return False

        task_id=res.json()["data"]['task_id']

        header = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"}
        
        while True:
            res = requests.get(f"https://mineru.net/api/v4/extract/task/{task_id}", headers=header)

            if res.json()["data"]['state']=='done':
                filepath=os.path.join(folder, f"{filename}.zip")
                response = requests.get(res.json()["data"]['full_zip_url'])
                with open(filepath, 'wb') as f:
                    f.write(response.content)       
                print(f"  [SUCCESS] 压缩包已保存到: {filepath}")
                return True

            elif res.json()["data"]['state']=='failed':
                print(f"  [FAIL] 解析失败 {url}")
                return False
                
            import time
            time.sleep(15)
            
            



