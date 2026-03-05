import logging
import os

from PIL import Image
from scientific_research_work.common.image import encode_image


def chart_squeezing(chart_path: str, quality: int = 70) -> None:
    try:
        with Image.open(chart_path) as img:
            original_width, original_height = img.size
            new_width = int(original_width * 0.7)
            new_height = int(original_height * 0.7)
            resized_img = img.resize((new_width, new_height), Image.LANCZOS)
            resized_img.save(chart_path, "JPEG", optimize=True, quality=quality)
    except Exception as exc:
        logging.error("chart_squeezing failed for %s: %s", chart_path, exc)


def data_dealing(save_result_path: str, start_num: int) -> None:
    for markdown_folder in sorted(os.listdir(save_result_path))[start_num:]:
        markdown_folder_path = os.path.join(save_result_path, markdown_folder)
        images_folder_path = os.path.join(markdown_folder_path, "images")

        with open(os.path.join(markdown_folder_path, f"{markdown_folder}.md"), "r", encoding="utf-8") as markdown_file:
            old_md_content = markdown_file.read()

        with open(os.path.join(markdown_folder_path, f"{markdown_folder}.md"), "w", encoding="utf-8") as markdown_file:
            all_image_name = os.listdir(images_folder_path)
            idx = 0
            for image_file in all_image_name:
                chart_squeezing(os.path.join(images_folder_path, image_file))

                if old_md_content.find(f"![](images/{image_file})") != -1:
                    new_name = f"image_{idx:03d}.jpg"
                    os.rename(
                        os.path.join(images_folder_path, image_file),
                        os.path.join(images_folder_path, new_name),
                    )
                    new_image_path = os.path.join(images_folder_path, new_name)
                    old_md_content = old_md_content.replace(f"![](images/{image_file})", f"![]({new_image_path})")
                    idx += 1
                else:
                    os.remove(os.path.join(images_folder_path, image_file))
            markdown_file.write(old_md_content)
