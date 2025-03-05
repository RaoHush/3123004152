import re
import sys
import glob
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def read_file(file_path):
    """强化异常处理的文件读取"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:  # 处理编码问题
            content = f.read().replace('\u3000', ' ').replace('\xa0', ' ')  # 替换异常空格
            return content
    except Exception as e:
        print(f"文件读取失败: {str(e)}")
        sys.exit(1)


def preprocess(text):
    """增强版预处理：清洗+分词+过滤"""
    # 清洗特殊字符
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text)
    # 结巴分词+过滤停用词
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
        print("示例: python main.py orig.txt orig_0.8_* output.txt")
        sys.exit(1)

    orig_path = sys.argv[1]
    plag_pattern = sys.argv[2]
    output_path = sys.argv[3]

    # 读取并预处理原文
    orig_processed = preprocess(read_file(orig_path))

    # 获取所有抄袭文件
    plag_files = sorted(glob.glob(plag_pattern))
    if not plag_files:
        print("未找到抄袭文件")
        sys.exit(1)

    # 批量处理
    results = []
    for plag_file in plag_files:
        # 处理每个抄袭文件
        plag_content = read_file(plag_file)
        plag_processed = preprocess(plag_content)

        # 计算相似度
        similarity = max(0.0, min(1.0, calc_similarity(orig_processed, plag_processed)))
        results.append(f"{similarity:.2f}")

    # 结果写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(results))


if __name__ == "__main__":
    main()