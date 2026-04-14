import json


with open('taylor_swift_corpus.json', 'r', encoding='utf-8') as f:
    data = json.load(f)


corpus_text = "\n".join(data.values())

with open('input.txt', 'w', encoding='utf-8') as f:
    f.write(corpus_text)

print(f"Extraction complete! Total characters: {len(corpus_text)}")