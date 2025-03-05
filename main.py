import re
import sys
import glob
import jieba
import os  # 新增导入os模块
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def read_file(file_path):
    """强化异常处理的文件读取"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read().replace('\u3000', ' ').replace('\xa0', ' ')
            return content
    except Exception as e:
        print(f"文件读取失败: {str(e)}")
        sys.exit(1)


def preprocess(text):
    """增强版预处理：清洗+分词+过滤"""
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text)
    words = [word for word in jieba.lcut(text) if len(word) > 1]
    return ' '.join(words)


def calc_similarity(orig_text, plag_text):
    """带异常处理的相似度计算"""
    try:
        vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b')
        matrix = vectorizer.fit_transform([orig_text, plag_text])
        return cosine_similarity(matrix[0], matrix[1])[0][0]
    except Exception as e:
        print(f"计算异常: {str(e)}")
        return 0.0


def main():
    if len(sys.argv) != 4:
        print("用法: python main.py <原文路径> <抄袭文件通配符> <输出路径>")
        sys.exit(1)

    orig_path = sys.argv[1]
    plag_pattern = sys.argv[2]
    output_path = sys.argv[3]

    orig_processed = preprocess(read_file(orig_path))
    plag_files = sorted(glob.glob(plag_pattern))

    if not plag_files:
        print("未找到抄袭文件")
        sys.exit(1)

    results = []
    for plag_file in plag_files:
        plag_content = read_file(plag_file)
        plag_processed = preprocess(plag_content)

        similarity = max(0.0, min(1.0, calc_similarity(orig_processed, plag_processed)))
        # 修改点：获取文件名并格式化输出
        filename = os.path.basename(plag_file)  # 提取纯文件名
        results.append(f"{filename}:{similarity:.2f}")  # 格式化为 文件名:评分

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(results))


if __name__ == "__main__":
    main()