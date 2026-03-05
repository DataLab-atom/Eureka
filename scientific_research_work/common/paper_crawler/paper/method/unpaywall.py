import requests
# from .. import filter_agent
# from dotenv import load_dotenv
# load_dotenv()  # 加载.env文件

from llm.env import get_env
UNPAYWALL_EMAIL = get_env("UNPAYWALL_EMAIL", "")

class Unpaywall:
    def __init__(self):
        pass

    @staticmethod
    def get_unpaywall_url_by_doi(doi):
        """尝试通过Unpaywall API查找开放获取的PDF链接"""
        try:
            email = UNPAYWALL_EMAIL
            api_url = f"https://api.unpaywall.org/v2/{doi}?email={email}" # 替换为你的邮箱
            response = requests.get(api_url, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data.get('best_oa_location') and data['best_oa_location'].get('url_for_pdf'):
                print(f"  [INFO] Unpaywall找到PDF链接: {data['best_oa_location']['url_for_pdf']}")
                return data['best_oa_location']['url_for_pdf']
        except requests.exceptions.RequestException as e:
            print(f"  [WARN] Unpaywall API 请求失败: {e}")
        except Exception as e:
            print(f"  [WARN] 解析Unpaywall响应失败: {e}")
        return None
    
    @staticmethod
    def get_unpaywall_search_result_by_title(query):
        """尝试通过Unpaywall API查找开放获取的PDF链接"""
        try:
            email = UNPAYWALL_EMAIL
            api_url = f"https://api.unpaywall.org/v2/search?query={query}&email={email}" # 替换为你的邮箱
            response = requests.get(api_url, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data
        except requests.exceptions.RequestException as e:
            print(f"  [WARN] Unpaywall API 请求失败: {e}")
        except Exception as e:
            print(f"  [WARN] 解析Unpaywall响应失败: {e}")
        return None
