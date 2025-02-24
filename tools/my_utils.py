import re
def sort_files(file_list):
    # 正则表达式提取文件名中的数字部分
    def extract_number(filename):
        match = re.search(r"-(\d+)\.png", filename)
        if match:
            return int(match.group(1))
        else:
            return float('inf')  # 如果没有匹配到数字，返回一个很大的数字以便排序到最后

    # 按照提取的数字部分排序文件名
    sorted_files = sorted(file_list, key=extract_number)
    return sorted_files