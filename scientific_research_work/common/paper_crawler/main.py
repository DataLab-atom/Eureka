import os
import time
import bibtexparser
from .paper.utils.download import Download
from .paper.utils.files import Files
from .paper.method.arxiv import Arxiv
# from paper.method.google_scholar import GoogleScholar
from .paper.method.unpaywall import Unpaywall
from .paper.filter_agent import FilterAgent
import logging
import bibtexparser
from bibtexparser.bibdatabase import BibDatabase
from bibtexparser.bwriter import BibTexWriter
from typing import Optional

# BIB_FILE_PATH = './reference.bib' # 54+
# DOWNLOAD_FOLDER = 'downloaded_papers'

def paper_crawler(BIB_FILE_PATH: str, DOWNLOAD_FOLDER: str, 
                  downloaded_paper_title_list: list, downloaded_paper_key_list: list,  
                  downloaded_paper_information: dict, downloaded_paper_information_bib: dict, 
                  api_parsing_method: str, all_paper_num: int,
                  batch_paper_id: list=None, batch_downloaded_paper_list: list=None, batch_paper_title_list: list=None):

    if not os.path.exists(DOWNLOAD_FOLDER):
        os.makedirs(DOWNLOAD_FOLDER)

    tmp_id=[] # 储存可以下载的paper id
    tmp_paper=[] # 储存可以下载的paper 

    try:
        with open(BIB_FILE_PATH, 'r', encoding='utf-8') as bibtex_file:
            bib_database = bibtexparser.load(bibtex_file)
    except FileNotFoundError:
        print(f"[ERROR] BibTeX 文件未找到: {BIB_FILE_PATH}")
        return
    except Exception as e:
        print(f"[ERROR] 解析 BibTeX 文件失败: {e}")
        return

    print(f"共找到 {len(bib_database.entries)} 个文献条目。")
    download_count = 0
    failed_downloads = []

    # 现在的论文编号从几开始
    start_num = len(downloaded_paper_title_list)

    for i, entry in enumerate(bib_database.entries):
        print(f"\n--- 处理条目 {i+1}/{len(bib_database.entries)}: {entry.get('ID', '未知ID')} ---")
        title = entry.get('title', '')
        authors_raw = entry.get('author', '')
        authors = [name.strip() for name in authors_raw.split(' and ')] if authors_raw else []
        doi = entry.get('doi')
        url_bib = entry.get('url') # BibTeX文件中的URL字段
        file_key = entry.get('ID', Files.sanitize_filename(title[:50] if title else "untitled"))

        print(f"  标题: {title[:70]}..." if title else "  标题: 未提供")
        print(f"  作者: {', '.join(authors[:2])}{' et al.' if len(authors) > 2 else ''}" if authors else "  作者: 未提供")
        print(f"  DOI: {doi}" if doi else "  DOI: 未提供")
        print(f"  URL (from bib): {url_bib}" if url_bib else "  URL (from bib): 未提供")

        pdf_found_and_downloaded = False

        if url_bib and url_bib.lower().endswith('.pdf'):
            print("  [ATTEMPT] 尝试BibTeX中的直接PDF链接...")
            if (api_parsing_method=='file') or (not api_parsing_method):
                if Download.download_pdf(url_bib, f"paper_{(start_num+download_count):03}", DOWNLOAD_FOLDER):
                    pdf_found_and_downloaded = True
            else:
                if Download.download_zip(url_bib, f"paper_{(start_num+download_count):03}", DOWNLOAD_FOLDER):
                    pdf_found_and_downloaded = True
                
        if not pdf_found_and_downloaded and (title or entry.get('elogging.info')):
            print("  [ATTEMPT] 尝试Arxiv搜索...")
            arxiv_pdf_url = Arxiv.search_arxiv_and_download(entry)
            if arxiv_pdf_url:
                if (api_parsing_method=='file') or (not api_parsing_method):
                    if Download.download_pdf(arxiv_pdf_url, f"paper_{(start_num+download_count):03}", DOWNLOAD_FOLDER):
                        pdf_found_and_downloaded = True
                else:
                    if Download.download_zip(arxiv_pdf_url, f"paper_{(start_num+download_count):03}", DOWNLOAD_FOLDER):
                        pdf_found_and_downloaded = True

        if not pdf_found_and_downloaded and doi:
            print("  [ATTEMPT] 尝试Unpaywall API...")
            unpaywall_pdf_url = Unpaywall.get_unpaywall_url_by_doi(doi)
            if unpaywall_pdf_url:
                if (api_parsing_method=='file') or (not api_parsing_method):
                    if Download.download_pdf(unpaywall_pdf_url, f"paper_{(start_num+download_count):03}", DOWNLOAD_FOLDER):
                        pdf_found_and_downloaded = True

                else:
                    if Download.download_zip(unpaywall_pdf_url, f"paper_{(start_num+download_count):03}", DOWNLOAD_FOLDER):
                        pdf_found_and_downloaded = True

            time.sleep(1)

        if not pdf_found_and_downloaded and title and not doi:
            import json

            print("  [ATTEMPT] 尝试Unpaywall API Title搜索...")
            filter_agent = FilterAgent()

            data = Unpaywall.get_unpaywall_search_result_by_title(title)

            res = json.loads(filter_agent.run(entry, data))
            if res['status']:
                if (api_parsing_method=='file') or (not api_parsing_method):
                    if Download.download_pdf(res['value'], f"paper_{(start_num+download_count):03}", DOWNLOAD_FOLDER):
                        pdf_found_and_downloaded = True
                else:
                    if Download.download_zip(res['value'], f"paper_{(start_num+download_count):03}", DOWNLOAD_FOLDER):
                        pdf_found_and_downloaded = True
            time.sleep(1)

        # if not pdf_found_and_downloaded and title and False:
        #     print("  [ATTEMPT] 尝试Google Scholar搜索...")
        #     scholar_pdf_url = GoogleScholar.search_google_scholar(title, authors)
        #     if scholar_pdf_url:
        #         if Download.download_pdf(scholar_pdf_url, title, DOWNLOAD_FOLDER):
        #             pdf_found_and_downloaded = True
        #     time.sleep(2)


        if pdf_found_and_downloaded:

            if batch_paper_title_list:
                downloaded_paper_title_list.append(batch_paper_title_list[i])
            else:
                downloaded_paper_title_list.append(title)
            
            downloaded_paper_key_list.append(file_key)
            
            if batch_paper_id:
                tmp_id.append(batch_paper_id[i])

            if batch_downloaded_paper_list:
                tmp_paper.append(batch_downloaded_paper_list[i])

            # 现已下载了的论文信息用于储存本地查看
            db = BibDatabase()
            db.entries = [entry]
            writer = BibTexWriter()
            downloaded_paper_information[f"paper_{(start_num+download_count):03}.pdf"]=writer.write(db)

            # 现已下载了的论文信息用于数据库的引用
            tmp_dict={}
        
            for key in entry.keys():
                if key in ['ENTRYTYPE', 'ID', 'author', 'title', 'year', 'volume', 'publisher', 'pages', 'number', 'journal', 'booktitle', 'organization', 'isbn']:
                    tmp_dict[key] = entry[key]

            db = BibDatabase()
            db.entries = [tmp_dict]
            writer = BibTexWriter()
            downloaded_paper_information_bib[f"paper_{(start_num+download_count):03}.pdf"]=writer.write(db)

            download_count += 1 # 成功下载计数+1

            if len(downloaded_paper_title_list) >= all_paper_num:
                print("论文下载数量已经达到预期！！！！！！！！")
                break

        else:
            print(f"  [FAIL] 未能下载论文: {file_key} ({title[:50]}...)")
            failed_downloads.append(f"{file_key} ({title[:50]}...) - DOI: {doi if doi else 'N/A'}")
            

        print("-" * 30)
        time.sleep(1) # 每个条目处理之间稍作停顿，避免过于频繁请求


    # 处理真正下载的paper id and paper
    if batch_paper_id:
        for idx,paper_id in enumerate(batch_paper_id):
            if paper_id not in tmp_id:
                del batch_paper_id[idx]

    if batch_downloaded_paper_list:
        for idx,paper in enumerate(batch_downloaded_paper_list):
            if paper not in tmp_paper:
                del batch_downloaded_paper_list[idx]


    print("\n--- 下载总结 ---")
    print(f"成功下载 {download_count} 篇论文。")
    if failed_downloads:
        print(f"未能下载 {len(failed_downloads)} 篇论文:")
        for item in failed_downloads:
            print(f"  - {item}")
        return
    else:
        print("所有可尝试的论文均已尝试下载。")
        return
