---
title: "A super small Vision-Language model with nanoVLM"
categories:
  - Blog
tags:
  - Artificial Intelligence
  - Multimodal AI
  - Large Language Models
authors:
  - me
layout: single
classes: wide
---

<div style="background-color: #fcfcfc; color: #2b2b2b; padding: 1rem; border-left: 4px solid #ccc; margin-bottom: 2rem;">
<p>
In the hope the community will benefit from this detailed post on the usage of nanoVLM for a specific use case.
</p>
</div>

I recently came accross a really interesting project from the community ([HuggingFace][hf]) that I found inspiring and easy to use: [nanoVLM][nanoVLM].
It provides a baseline in pure pytorch to train, evaluate and develop small Vision-Language model.

The desire to play with this project comes from another long-term personal project that started more or less 10 years ago.
What I need to do? I have to solve a Vision Question Answering problem on a dataset that is really super small.

The dataset has more or less 3000 items (image, text) and the purpose is VQA with always the same question from the user and a long - 10 to 50 words - answer from the assistant, let me show an example:

** dataset entry example

And yes, this VQA problem has to deal with factual informations. I know the scope of the dataset is limited but this is a different story.

Environment details:

* [training] a gpu with 12GB or RAM (fortunately)
* [deployment] limited computational resources: no datacenters, no gpus, just one cpu and 2 GB of rams.

Implementation details:

* I made the dataset compatible with the repo by simply following the signature of the [cauldron][cauldron]:

```python

from datasets import Dataset, Features, Value, Image as HFImage

features = Features({
        "images": Sequence(feature=HFImage()),
        "texts": [{
            "user": Value("string"),
            "assistant": Value("string"),
            "source": Value("string"),
        }]
    })
```


* given the point above, I can't even afford with the standard parameters (config.py)
* therefore I made these major changes to the LLM and VIT parameters:

```python
@dataclass
class VLMConfig:
  ...
  vit_n_heads: int = 1
  vit_n_blocks: int = 1
  ...
  lm_n_heads: int = 1
  lm_n_kv_heads: int = 1
  lm_n_blocks: int = 1


@dataclass
class TrainConfig:
  ...
  batch_size: int = 60

```

<h2> First fail </h2>

Ready to train...wait for 6 hours while observing the validation loss decreasing!
Happy to have obtained something, try to generate (generate.py): same phrases, no changes...for sure overfitting!

<h2> Increased dataset </h2>

I asked the community for suggestions:

<ul>
  <li>the overfitting is a good signal of the capability of the model even the above config.py</li>
  <li>increase the dataset because 3k items is not enough to hope for kind of generalization signals</li>
</ul>

How to increase the dataset?

<b>First phase</b>: synthetic generation. I used Ollama and wizardlm2_7b to generate many versions (more or less 10) of every entry in the original dataset.
An example:

So with this first extension now the dataset is x10 times to 30k.

Can I do more? I fear the data redudancy introduced with synthetic generation can make the model collapse with no chance to generalize.

<b>The simple idea</b>: to extend the original dataset with the_cauldron (or portions of it).
I searched for portions that are semantically similar to my base dataset and it comes out that "localized_narratives" with 200k more items is a good choice.

<h2> Training trials and results </h2>

I spent more or less one week on training many configurations based on the major choices showed above.

Table 1. shows the differences among the training iterations performed.

Now I have a dataset of 230k items, for sure a better starting point!

<table border="1" style="border-collapse: collapse; text-align: center;">
  <thead>
    <tr>
      <th>iteration super_small_xx</th><th>3</th><th>4</th><th>5</th><th>6</th><th>7</th><th>8</th><th>9</th><th>10</th><th>11</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>lm_hidden_dim</td><td>72</td><td>72</td><td>72</td><td>144</td><td>108</td><td>108</td><td>144</td><td>144</td><td>216</td></tr>
    <tr><td>lm_inter_dim</td><td>192</td><td>192</td><td>192</td><td>192</td><td>192</td><td>96</td><td>192</td><td>192</td><td>192</td></tr>
    <tr><td>lm_n_blocks</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td></tr>
    <tr><td>lm_n_heads</td><td>4</td><td>6</td><td>6</td><td>6</td><td>6</td><td>6</td><td>6</td><td>6</td><td>6</td></tr>
    <tr><td>lm_n_kv_heads</td><td>1</td><td>1</td><td>3</td><td>2</td><td>2</td><td>1</td><td>1</td><td>1</td><td>1</td></tr>
    <tr><td>vit_inter_dim</td><td>3072</td><td>3072</td><td>3072</td><td>3072</td><td>3072</td><td>3072</td><td>1536</td><td>768</td><td>768</td></tr>
    <tr><td>vit_n_blocks</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td></tr>
    <tr><td>vit_n_heads</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td></tr>
    <tr><td>LLM SIZE (#PARAMETERS)</td><td>3,593,592</td><td>3,593,592</td><td>3,594,456</td><td>7,216,560</td><td>5,402,052</td><td>5,367,060</td><td>7,209,648</td><td>7,209,648</td><td>10,850,760</td></tr>
    <tr><td>VIT SIZE (#PARAMETERS)</td><td>7,830,528</td><td>7,830,528</td><td>7,830,528</td><td>7,830,528</td><td>7,830,528</td><td>7,830,528</td><td>5,469,696</td><td>4,289,280</td><td>4,289,280</td></tr>
  </tbody>
</table>

<div style="margin-bottom: 1.5rem;">
  <div class="text-center" style="color: #646769;font-size: 0.75em;margin-left:5rem;margin-right:5rem">Table 1: config.py parameter changes for all training iterations and model parameter size (LLM and Vision tower).</div>
</div>

Figure 1 illustrates the progression of the validation loss <b>val_loss</b>, which I used as main indicator instead of mmstar.

<div style="margin-bottom: 1.5rem;">
  <img src="{{ site.baseurl }}/assets/images/W&B Chart 25_07_2025, 23_25_51.svg" alt="Validation loss during training">
  <div class="text-center" style="color: #646769;font-size: 0.75em;margin-left:5rem;margin-right:5rem">Figure 1: Validation loss trend over steps for all training iterations. Y axes values are clipped between [1.7, 2.5] for ease of visualization.</div>
</div>

Figure 2. shows the token per second during training which is usefull to discover correlation with the number of model parameters.

<div style="margin-bottom: 1.5rem;">
  <img src="{{ site.baseurl }}/assets/images/W&B Chart 25_07_2025, 23_17_58.svg" alt="Tokens per second during training">
  <div class="text-center" style="color: #646769;font-size: 0.75em;margin-left:5rem;margin-right:5rem">Figure 2: Tokens per second over steps for all training iterations.</div>
</div>

The generation phase (examples)

<h2> Conclusions </h2>



[nanoVLM]: https://github.com/huggingface/nanoVLM
[hf]: https://huggingface.co
[cauldron]: https://huggingface.co/datasets/HuggingFaceM4/the_cauldron

<hr/>

<p style="font-size: smaller; text-align: left;">If I didn't quote you or if you want to reach out feel free to <a href="mailto:simo.brazzo@gmail.com">contact me</a>.</p>
<p style="font-size: smaller; text-align: left;">© [Simone Brazzo] [2025] - Licensed under <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>  with the following additional restriction: this content can be only used to train open-source AI models, where training data, models weights, architectures and training procedures are publicly available.</p>

<hr/>