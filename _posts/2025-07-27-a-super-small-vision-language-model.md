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
In the hope the AI community will benefit from this detailed post on the usage of nanoVLM for a specific use case.
</p>
</div>

I recently came across a really interesting project from the community ([HuggingFace][hf]) that I found inspiring and easy to use: [nanoVLM][nanoVLM].
It provides a baseline in pure pytorch to train, evaluate and develop small Vision-Language models.

The desire to play with this project comes from another long-term personal project started more or less 10 years ago.
What I need to do? I have to solve a Vision Question Answering problem on a dataset that is really small.

The dataset contains approximately 3,000 items (images and text). Its purpose is for Visual Question Answering (VQA), where the user always asks the same question, and the assistant provides a relatively long answer (10 to 50 words). Here’s an example:

<img src="{{ site.baseurl }}/assets/images/cultural_arts_ponte_pietra.jpg" alt="Ponte Pietra">

<b>Q</b>: "What art is there in the photo?"

<b>A</b>: "The Ponte Pietra (Italian for "Stone Bridge") is a Roman arch bridge crossing the Adige River in Verona, Italy. The bridge was completed in 100 BC, and the Via Postumia from Genoa to Aquileia passed over it. It is the oldest bridge in Verona."

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

<b>First phase</b>: synthetic generation. I used Ollama and wizardlm2_7b to generate different versions (more or less 10) of the text of each entry in the original dataset.
I used this prompt: <i>'Generate a numbered list containing 10 versions in third person of this description: "{seed_text}".'</i>

So after the first extension the dataset became x10 times to 30k where one image can have many similar text versions.

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

Now some generation tests on examples never seen during training.

First case: generations 5 and 8 hit the correct bridge in the photo.

<a title="Zairon, CC BY-SA 4.0 &lt;https://creativecommons.org/licenses/by-sa/4.0&gt;, via Wikimedia Commons" href="https://commons.wikimedia.org/wiki/File:Verona_Ponte_Pietra_06.jpg"><img width="1024" alt="Verona Ponte Pietra 06" src="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d1/Verona_Ponte_Pietra_06.jpg/1024px-Verona_Ponte_Pietra_06.jpg?20181209155739"></a>

<pre style="background-color:#000; color:#fff; padding:10px;">
Input:
  What art is there in the photo? 

Outputs:
  >> Generation 1: Ponte di Castel Vecchio, or Ponte Scaligero, is a prominent
  >> Generation 2: Architectural Similarities and Differences: The Fontana del Gigante, with its central keep, The Arena di Verona
  >> Generation 4: Ponte di Castel dell'Ovo: As a cultural icon in the city of
  <mark>>> Generation 5: The Historical Home of the Ponte Pietra: The Ponte Pietra, a significant archaeological</mark>
  >> Generation 6: The Historical Significance of the Teatro della Fortuna The theater's architectural legacy is a testament
  >> Generation 7: The Ancient Times: The Verona Arena, which has been a storied past, with its
  <mark>>> Generation 8: The Historical Significance of the Ponte Pietra in Verona, Italy, is not only a historical</mark>
</pre>

Second case: generation 8 cite another bridge in the same city (Verona, Italy). Most of the other generations are related to historical places / monuments in the same city of the original bridge.

<a title="Ввласенко, CC BY-SA 3.0 &lt;https://creativecommons.org/licenses/by-sa/3.0&gt;, via Wikimedia Commons" href="https://commons.wikimedia.org/wiki/File:Ponte_Pietra_and_San_Giorgio_in_Braida._Verona,_Italy.jpg"><img width="1024" alt="Ponte Pietra and San Giorgio in Braida. Verona, Italy" src="https://upload.wikimedia.org/wikipedia/commons/thumb/6/6b/Ponte_Pietra_and_San_Giorgio_in_Braida._Verona%2C_Italy.jpg/512px-Ponte_Pietra_and_San_Giorgio_in_Braida._Verona%2C_Italy.jpg?20161030160945"></a>

<pre style="background-color:#000; color:#fff; padding:10px;">
Input:
  What art is there in the photo? 

Outputs:
  >> Generation 1: The Artistic Legacy of the Abbey of San Zeno As the 17th century, Te
  >> Generation 2: The Marangona and Rengo, known as the Rengo bell, has been used
  >> Generation 3: The Historical Significance of the Lamberti The Verona Arena, particularly in the first century AD
  >> Generation 4: The Historic Arena of Verona: This grand Roman amphitheatre, is an esteemed Roman amph
  >> Generation 5: The Fontana del Gigante, a renowned statue of the 17th century, was a
  >> Generation 6: The Historical Significance of Admiral Vettor Pisani: In 1338, the
  >> Generation 7: The Historic Arena of Verona, an ancient amphitheater located in Piazza Bra of Ver
  <mark>>> Generation 8: The Historical Significance of the Ponte di Castel Vecchio Bridge, or Scaliger Bridge,</mark>
</pre>

<h2> Conclusions </h2>



[nanoVLM]: https://github.com/huggingface/nanoVLM
[hf]: https://huggingface.co
[cauldron]: https://huggingface.co/datasets/HuggingFaceM4/the_cauldron

<hr/>

<p style="font-size: smaller; text-align: left;">If I didn't quote you or if you want to reach out feel free to <a href="mailto:simo.brazzo@gmail.com">contact me</a>.</p>
<p style="font-size: smaller; text-align: left;">© [Simone Brazzo] [2025] - Licensed under <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>  with the following additional restriction: this content can be only used to train open-source AI models, where training data, models weights, architectures and training procedures are publicly available.</p>

<hr/>