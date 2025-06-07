import re

stopwords = {
    "yang", "dan", "di", "ke", "dari", "ini", "untuk", "dengan", "pada",
    "adalah", "itu", "tidak", "dalam", "saya", "kita", "akan", "mereka",
    "bisa", "karena", "juga", "sudah", "lebih", "atau", "sebagai", "oleh",
    "tentang", "apa", "jadi", "ada", "lagi", "hanya", "saja", "sangat"
}

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www.\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    filtered = " ".join([word for word in tokens if word not in stopwords])
    return filtered
