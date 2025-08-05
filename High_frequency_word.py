import fitz  # PyMuPDF
import re
import pandas as pd
import numpy as np

# 打开 PDF 文件
doc = fitz.open("highfreqwords.pdf")

# 提取所有页面的文本
text = ""
for page in doc:
    text += page.get_text()

# 正则匹配：word, number1, number2 （第二列频率不使用）
pattern = re.findall(r"\b([a-zA-Z]+),\s*(\d+),\s*\d+", text)

# 构建 DataFrame（保留所有词汇）
word_freq = [(word.lower(), int(freq)) for word, freq in pattern]

df = pd.DataFrame(word_freq, columns=["word", "frequency"])
df = df.drop_duplicates(subset="word")  # 去重（同一个词可能在不同位置重复）

# 保存为 CSV 文件
df.to_csv("highfreqwords.csv", index=False)

print("✅ 所有词汇已提取并保存为 highfreqwords.csv")

