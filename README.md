# Integrated-Gradients
The PyTorch code for integrated gradients in [paper](https://arxiv.org/abs/1703.01365).

The original implementation is in tensorflow and the captum's code is kind of heavy. Thus this repository comes.

The code here can be very simple for understanding the theorem and extending to other understream tasks.

# How to Use
Here is an example for sentiment classification in NLP (in sentiment_classification.ipynb). Note this code must be on the Jupyter Notebook for its visualization depends on the `Ipython`.

```python
# for sentiment classification
pretrain_path = "https://huggingface.co/fabriceyhc/bert-base-uncased-imdb/tree/main"
bert_config = BertConfig.from_pretrained(pretrain_path)
bert_config.output_hidden_states = True
model = BertForSequenceClassification.from_pretrained(pretrain_path, config=bert_config).cuda()

tk = BertTokenizer.from_pretrained(pretrain_path)
input_text = ["I am happy because the weather is extremely good!",
              "This film is bad! I hate it!"]
inputs = tk(input_text, max_length=128, return_tensors="pt", truncation=True, padding=True)
inputs = {k: v.cuda() for k, v in inputs.items()}
grads, labels = get_integrated_gradients(model, **inputs)
scores = get_scores(grads)
tok_text = [tk.tokenize(t) for t in input_text]
# get the top-3 strong and weak correlation
positives, negatives, tok_text = get_related(tok_text, scores, 3)
# red for strong correlation
# green for weak correlation
visualize(tok_text, positives, negatives, labels)
```
The result should be:

![](/example.png)