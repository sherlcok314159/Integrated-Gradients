import torch
from copy import deepcopy
from torch.autograd import grad
from einops import reduce, rearrange
from IPython.display import display, HTML

# functions
def get_scores(x):
    scores = torch.sqrt((x ** 2).sum(-1))
    max_s, min_s = scores.max(1, True).values, scores.min(1, True).values
    return (scores - min_s) / (max_s - min_s)

def get_integrated_gradients(
    model, 
    input_ids, 
    token_type_ids, 
    attention_mask, 
    baseline=None, 
    num_steps=50
):
    # get the word embedding matrix
    input_embed = model.bert.embeddings.word_embeddings.weight
    backup_embed = deepcopy(input_embed.data)
    if baseline is None:
        baseline = torch.zeros_like(backup_embed)

    grads = []
    for step in range(num_steps + 1):
        alpha = step / num_steps
        input_embed.data = baseline + alpha * (backup_embed - baseline)
        bert_outputs = model(input_ids, attention_mask, token_type_ids)
        logits, hidden_states = bert_outputs.logits, bert_outputs.hidden_states
        logits_l = reduce(logits, "b h -> b", reduction="max")
        # we calculate the derivates of the output of BertEmbedding
        embed_out = hidden_states[0]
        g = grad(logits_l, embed_out, grad_outputs=torch.ones_like(logits_l))[0]
        grads.append(g)

    labels = logits.argmax(dim=1).tolist()
    grads = rearrange(grads, "n b h m -> n b h m")
    grads = (grads[:-1] + grads[1:]) / 2.
    avg_grads = grads.mean(0) 

    integrated_grads = embed_out * avg_grads
    return integrated_grads, labels

def get_related(tok_text, scores, n=1):
    scores = scores.tolist()
    postives, negatives = [], []
    for (s, text) in zip(scores, tok_text):
        # remove [CLS] & [SEP] & [PAD]
        s = s[1: len(text) + 1]
        # n should no more than half of s
        half = len(s) // 2
        idx = n if n <= half else half
        s = sorted(enumerate(s), key=lambda x: x[1], reverse=True)
        s = [i[0] for i in s]
        highs = [text[idx] for idx in s[:idx]]
        lows = [text[idx] for idx in s[-idx:]]
        postives.append(highs)
        negatives.append(lows)
    return postives, negatives, tok_text

show = lambda s: display(HTML(f"{s}"))

set_color = lambda x, y: "<style>" + f".{x}" + "{color: rgb" + f"({y[0]}, {y[1]}, {y[2]})" + ";}</style>"

def add_show(cls, idx, show_str, text, degree=60):
    # red for strong correlation
    # green for weak correlation
    space = "&nbsp;" * 3
    name = f"{cls}{idx}"
    if cls == "neg":
        pattern = [0, 255 - idx * degree, 0]
    elif cls == "pos":
        pattern = [255 - idx * degree, 0, 0]
    else:
        pattern = [255, 255, 255]

    show_str += set_color(f"{name}", pattern) + f"<a class={name}>{text}</a>" + space
    return show_str

def find_idx(lis, t):
    for (i, j) in enumerate(lis):
        if j == t:
            return i
    return -1

def visualize(tok_text, positives, negatives, labels, degree=65):
    for (text, pos, neg, label) in zip(tok_text, positives, negatives, labels):
        show_str = ""
        for t in text:
            pos_idx, neg_idx = find_idx(pos, t), find_idx(neg, t) 
            if neg_idx != -1:
                show_str = add_show("neg", neg_idx, show_str, t, degree) 
            elif pos_idx != -1:
                show_str = add_show("pos", pos_idx, show_str, t, degree) 
            else:
                show_str = add_show("orig", 0, show_str, t, degree)

        show_str +=  f"<a class=orig0>LABEL ==> {label}</a>"
        show(show_str)
        show_str = ""