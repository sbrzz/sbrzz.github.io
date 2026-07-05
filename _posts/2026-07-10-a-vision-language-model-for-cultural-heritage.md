---
title: "Towards a Vision-Language Model for Cultural Heritage"
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
words_per_minute: 250
---

<!-- 
<div style="background-color: #fcfcfc; color: #2b2b2b; padding: 1rem; border-left: 4px solid #ccc; margin-bottom: 2rem;">
<p>
In the hope the AI community will benefit from this small post on the usage of nanoVLM for a specific use case.
</p>
</div> -->

Almost one year ago I wrote a [post][gradientszonelink] on the usage of [nanoVLM][nanoVLM] for a specific use case.
That research was focused on the trainability of a modified (really small in number of parameters) Vision-Language Model (VLM) for a specific domain.

That experience took me to another (higher) level:: build an efficient and effective VLM for [cultural-arts.com][cultural-arts], an open-source project dedicated to promoting art, culture, and education in human cultural heritage.

<h2> Implementation details  </h2>

To summarize the major changes:

* I take advantage of the new [FineVision][fine-vision] dataset. In particular my interest was in the Google Landmarks subset (200k rows) which sounds semantically appropriate for the target topic.
* I used the open-source [cultural-arts.com][cultural-arts] dataset adapted in terms of signature to be compatible with the data flow.
* Major changes for the vision tower and llm backbones:

```python
@dataclass
class VLMConfig:
  ...
  vit_model_type: str = 'google/siglip-base-patch16-224'
  ...


@dataclass
class TrainConfig:
  ...
  lm_model_type: str = 'HuggingFaceTB/SmolLM2-135M'
  ...

```

<h2>Training trials</h2>

The trainings trials was fortunately quite good since the beginning.

Figure 1 shows a detail that I found really interesting: the validation loss of a training in the current setup (lowest line) with respect to the experiments (grouped lines) of one year ago. I justify this behaviour with the usage of unmodified backbones (Vision Tower and LLM), characteristic not valid for the old experiments.  

<div style="margin-bottom: 1.5rem;">
  <img src="{{ site.baseurl }}/assets/images/W&B Chart 04_07_2026, 22_35_01.svg" alt="Validation loss during training">
  <div class="text-center" style="color: #646769;font-size: 0.75em;margin-left:5rem;margin-right:5rem">Figure 1: Comparison of validation loss between current models (lower line) and the ones trained one year ago.</div>
</div>

<h2>Test scenario</h2>

While one year ago the focus was on generating good signals from the  model, i.e. to generate content even in minimum part related to the input photo, this time the goal is different: to measure the model effectiveness.

For that reason I take a portion of the [cultural-arts.com][cultural-arts] dataset, which encompass a place really familiar to me: Prato della Valle (overview in Figure 1). Other than being a place in my city of birth, Prato della Valle is one of the largest squares in the EU (88620mq) and it is really beautiful.

<div style="margin-bottom: 1.5rem;">
  <img src="{{ site.baseurl }}/assets/images/1280px-Prato_della_Valle.jpg" alt="Prato Della Valle">
  <div class="text-center" style="color: #646769;font-size: 0.75em;margin-left:5rem;margin-right:5rem">
    Figure 1: Prato della Valle. Hannelore, <a href="https://creativecommons.org/licenses/by-sa/3.0" target="_blank" rel="noopener">CC BY-SA 3.0</a>, via <a href="https://commons.wikimedia.org/wiki/File:Prato_della_Valle.JPG" target="_blank" rel="noopener">Wikimedia Commons</a>
  </div>
</div>

From Figure 1 you can see the statues along the waterway, and if you are a courious tourist you may ask: who are they?
For that reason I used the trained VLM as a main motor for an app that takes as input a photo and a request, i.e. "describe this image"!

In Figure 2 some examples of images from the test dataset.

<div style="margin-bottom: 1.5rem;">
  <img src="{{ site.baseurl }}/assets/images/merged_statues.jpg" alt="Merged Statues">
  <div class="text-center" style="color: #646769;font-size: 0.75em;margin-left:5rem;margin-right:5rem">
    Figure 2: images from cultural-arts.com dataset. From left to right: Andrea Mantegna, Galileo Galilei  and Ludovico Ariosto with a funny pigeon on the head.
  </div>
</div>

What can be done to understand if the generated content is meaninful? A really simple metric I tought to use is checking if the final output contains the subject name (like Galileo Galilei). I call this metric <b>in-topic accuracy</b> since it is measured like, an accuracy :eyes:.

The <b>in-topic accuracy</b> is about 83% for a greedy generation approach, while 80% when I used stochasticity. That measure is computed on about 2000 images of the statues, taken from different angles and environmental conditions. Really encouraging results!

<h2> Conclusions </h2>

TODO

[nanoVLM]: https://github.com/huggingface/nanoVLM
[hf]: https://huggingface.co
[cauldron]: https://huggingface.co/datasets/HuggingFaceM4/the_cauldron
[gradientszonelink]: https://www.gradients.zone/blog/a-super-small-vision-language-model/
[cultural-arts]: https://cultural-arts.com/
[fine-vision]: https://huggingface.co/datasets/HuggingFaceM4/FineVision
[mantegna]: https://it.wikipedia.org/wiki/Andrea_Mantegna
[galilei]: https://it.wikipedia.org/wiki/Galileo_Galilei
[ariosto]: https://it.wikipedia.org/wiki/Ludovico_Ariosto

<hr/>

<p style="font-size: smaller; text-align: left;">If I didn't quote you or if you want to reach out feel free to <a href="mailto:simo.brazzo@gmail.com">contact me</a>.</p>
<p style="font-size: smaller; text-align: left;">© [Simone Brazzo]  - Licensed under <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>  with the following additional restriction: this content can be only used to train open-source AI models, where training data, models weights, architectures and training procedures are publicly available.</p>

<hr/>