# 日本語医療固有表現抽出器

## 概要

ソーシャル・コンピューティング研究室さまより公開されている[MedTxt-CR](https://sociocom.naist.jp/medtxt/cr/)を用いて、alabniiさまより公開されている[RoBERTa](https://huggingface.co/alabnii/jmedroberta-base-sentencepiece-vocab50000)をfine-tuningした固有表現抽出モデルです。

## fine-tuning時のハイパーパラメータ

- learning rate: 1e-5
- batch size: 48
- optimizer: AdamW
- scheduler: linear
- epochs: 67
- max seq: 500

## 使用方法

入力は文単位で出力はIOB2系列です。

### 1

```python
from transformers import BertForTokenClassification, AutoModel, AutoTokenizer
import mojimoji
import torch
text = "サンプルテキスト"
model_name = "daisaku-s/med_ner"
with torch.inference_mode():
    model = BertForTokenClassification.from_pretrained(model_name).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    idx2tag = model.config.id2label
    vecs = tokenizer(mojimoji.han_to_zen(text), 
                     padding=True, 
                     truncation=True, 
                     return_tensors="pt")
    ner_logits = model(input_ids=vecs["input_ids"], 
                       attention_mask=vecs["attention_mask"])
    idx = torch.argmax(ner_logits.logits, dim=2).detach().cpu().numpy().tolist()[0]
    token = [tokenizer.convert_ids_to_tokens(v) for v in vecs["input_ids"]][0][1:-1]
    pred_tag = [idx2tag[x] for x in idx][1:-1]
```


固有表現とタグのペアで出力する処理

```python
from allennlp.data.dataset_readers.dataset_utils import span_utils
spans = span_utils.bio_tags_to_spans(pred_tag)
for span in spans:
    tag = span[0]
    ent = "".join(token[span[1][0]: span[1][1]+1])
    print("Entity: {0} Tag: {1}".format(ent, tag))
```

### 2

```python
from transformers import pipeline
text = "サンプルテキスト"
model_name = "daisaku-s/med_ner"
ner = pipeline("ner", model=model_name)
results = ner(text)
print(results)
```



## 実践的な使用例

### 1

### 2

## 実験結果 (Micro-F1)

5分割交差検証による内挿評価の結果になります。
訓練データの20%を検証データとして使用し、100エポック学習させた中で検証データにおけるMicro-F1が最も高かった時のエポック時のモデルを用いてテストデータの評価を行いました。
なお、検証データにおける最適なエポックの平均値で上記のモデルは学習しております。

|Fold|RoBERTa|
|:---|---:|
|0 |0.714|
|1 |0.710|
|2 |0.736|
|3 |0.716|
|4 |0.736|
|Avg. |0.722|

## 文献
- [MedTxt-CR: 症例報告 (Case Reports) コーパス](https://sociocom.naist.jp/medtxt/cr/)
- 杉本海人, 壹岐太一, 知田悠生, 金沢輝一, 相澤彰子, JMedRoBERTa: 日本語の医学論文にもとづいた事前学習済み言語モデルの構築と評価, 言語処理学会第29回年次大会, 2023.
