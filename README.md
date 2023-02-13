# MEDTXT-NER

## 概要

ソーシャル・コンピューティング研究室より公開されている[MedTxt-CR](https://sociocom.naist.jp/medtxt/cr/)を用いて、alabniiより公開されている[RoBERTa](https://huggingface.co/alabnii/jmedroberta-base-sentencepiece-vocab50000)を固有表現抽出用にfine-tuningしたモデルです。

## 使い方

```python
from transformers import BertForTokenClassification, AutoModel, AutoTokenizer
import mojimoji
text = "サンプルテキスト"
with torch.inference_mode():
    model = BertForTokenClassification.from_pretrained("daisaku-s/med_ner").eval()
    tokenizer = AutoTokenizer.from_pretrained("daisaku-s/med_ner")
    idx2tag = model.config.id2label
    vecs = tokenizer(mojimoji.han_to_zen(text), 
                     padding=True, 
                     truncation=True, 
                     return_tensors="pt")
    ner_logits = model(input_ids=vecs["input_ids"], 
                       attention_mask=vecs["attention_mask"])
    idx = torch.argmax(ner_logits.logits, dim=2).detach().cpu().numpy().tolist()[0]
    pred_tag = [idx2tag[x] for x in idx][1:-1]
```

## 実験結果 (Micro-F1)

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
- 杉本海人, 壹岐太一, 知田悠生, 金沢輝一, 相澤彰子, JMedRoBERTa: 日本語の医学論文にもとづいた事前

