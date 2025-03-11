---
title: "Are GPTs STILL GPTs?"
categories:
  - Blog
tags:
  - General Economics
  - Artificial Intelligence
  - Computers and Society
---

About one year ago I read a research [paper][gptsaregpts] about AI - specifically Large Language Models (LLM) - and its intersection with economics and the labor market (US). I was interested in a specific topic discussed by the authors - most of them from OpenAI - regarding the fact that Generative Pretrained Transformers (GPTs), a prominent type of LLM, exhibit traits of General Purpose Technologies (yes, still GPTs).
<br>
What is a General Purpose Technology? Well, it’s a technology that must meet these three criterias:

- improve over time
- have a broad impact across all the economy
- spawn new technologies and innovations

Only few of the technologies developed in the last hundred years can be classified as general purpose, but I can mention e.g. the steam engine, electricity and information technology.
<br>
<br>
About point 2 and one year and half after the paper release, I think the economy is still absorbing the potential of LLMs and in general AI. I say “I think” but in reality it is more than just a simple feeling. Among various market research companies, [Statista][statista] foresees the global AI market to grow from USD 243.70 billion in 2025 to USD 826.70 billion by 2030, with a compound annual growth rate of 27.67%.
<br>
<br>
Point 3 was deeply discussed in the paper, did we see something in the meanwhile regarding it? Yes of course! Do you remember the robot from [Figure][figureai]? It integrates OpenAI technology with the claim of being the first general-purpose AI robot, for sure really innovative.

What about point 1 “improving over time”? Still improving and how?

This post is mainly focused on this point. Why this stupid question, you may think. Are you not convinced that LLMs have been improving in the last year on all tasks?

Well, let’s have a quick overview.

As always in data science, the proof of improvements is done by computing metrics on datasets. Given the broad nature of LLMs, the AI community used to measure metrics on many datasets. For example we have:

- [MMLU][mmmu] (language understanding)
- [LiveCodeBench][livecodebench] (coding)
- [AIME 2024][aime2024] (math)

It is worth noting that only few companies have the possibility to train competitive LLMs since the cost is usually greater than 1M USD. 
<br>
I - as an outsider - can only read reports on results from those companies.
Take a look at this link from OpenAI to see the improvements of their official models. Just to summarize the mean score among various [datasets][datasetsres]:

<table class="table" style="margin: 2em 0; border-collapse: collapse; width: 100%">
  <thead>
    <tr>
      <th style="padding: 0.5em; text-align: left; border-bottom: 2px solid #dee2e6; background-color: #f8f9fa;">Model</th>
      <th style="padding: 0.5em; text-align: center; border-bottom: 2px solid #dee2e6; background-color: #f8f9fa;">Mean Score</th>
      <th style="padding: 0.5em; text-align: left; border-bottom: 2px solid #dee2e6; background-color: #f8f9fa;">Release Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding: 0.5em; border-bottom: 1px solid #dee2e6;">GPT-4o</td>
      <td style="padding: 0.5em; border-bottom: 1px solid #dee2e6; text-align: center;">52.94</td>
      <td style="padding: 0.5em; border-bottom: 1px solid #dee2e6;">May 13, 2024</td>
    </tr>
    <tr>
      <td style="padding: 0.5em; border-bottom: 1px solid #dee2e6;">o1</td>
      <td style="padding: 0.5em; border-bottom: 1px solid #dee2e6; text-align: center;">80.83</td>
      <td style="padding: 0.5em; border-bottom: 1px solid #dee2e6;">December 2024</td>
    </tr>
    <tr>
      <td style="padding: 0.5em; border-bottom: 1px solid #dee2e6;">o3-mini-high</td>
      <td style="padding: 0.5em; border-bottom: 1px solid #dee2e6; text-align: center;">86.90</td>
      <td style="padding: 0.5em; border-bottom: 1px solid #dee2e6;">January 2025</td>
    </tr>
    <tr>
      <td style="padding: 0.5em; border-bottom: 1px solid #dee2e6;">o3-mini4</td>
      <td style="padding: 0.5em; border-bottom: 1px solid #dee2e6; text-align: center;">85.90</td>
      <td style="padding: 0.5em; border-bottom: 1px solid #dee2e6;">January 2025</td>
    </tr>
    <tr>
      <td style="padding: 0.5em; border-bottom: 1px solid #dee2e6;">o3-mini-low</td>
      <td style="padding: 0.5em; border-bottom: 1px solid #dee2e6; text-align: center;">84.90</td>
      <td style="padding: 0.5em; border-bottom: 1px solid #dee2e6;">January 2025</td>
    </tr>
  </tbody>
</table>


In less than one year, the improvements are stunning (~+30%).

Would you also like to know the point of view of a top-5 AI expert?
We can [listen][bengio] to Dr. Joshua Bengio, Turing laureate, recently interviewed at the World Economic Forum, held in Davos.
<br>
Therefore, Dr. Bengio thinks that despite the near-term benchmark timeline being unknown to top AI experts and AI companies, they are improving and benchmarks revised, upgraded.

Once discussed the “if”, let's discuss the “how”.

I would like to highlight a recent [statement][ilya] (NeurIPS 2024) by another world-class expert, Ilya Suskever: “pretraining LLMs as we know today is finished, we have but one internet”.
<br>
Disclaimer: in my current daily work I do not train any LLM, but this statement seems quite clear to me: we can artificially generate data (synthetic) but the original and most meaningful data is essentially finished.

Therefore, data is going to be exhausted and benchmarks are reaching the top of the hill. Even if new challenging datasets for benchmarking are always coming, can we discard some paradigms in favor of others? 

Can we soften e.g. the [scaling law][scalinglaw]? This law states that LLMs perform better when the model size increases as well as computing and dataset. However, I remark on the phrase "we have but one internet".
<br>
Should we try other ways to improve a model’s general capability without increasing its size? New techniques e.g. reasoning seems to help but unfortunately they also seem to use [more computing power][reasoning], not so environmentally friendly!

I am not a great fan of “bigger is better”, but I would say that another emergent paradigm is “smaller and specialized”.
Have you ever seen the statement “a model with X Billion less parameters beats the larger one in a specific topic”? It is recurrent at [Hugging Face][hg] and in the AI community.

In my - not alone - opinion, models (not only GenAI) are going to be even more specialized. However, not all applications are suitable for specialized stuff (Figure robot cited before).

Specialized => smaller => cheaper => large - economic - adoption?

Finally, are GPTs STILL GPTs? No reasons to say no.

<hr/>

<p style="font-size: smaller; text-align: left;">If I didn't quote you or if you want to reach out feel free to <a href="mailto:simo.brazzo@gmail.com">contact me</a>.</p>
<p style="font-size: smaller; text-align: left;">© [Simone Brazzo] [2025] - Licensed under <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>  with the following additional restriction: this content can be only used to train open-source AI models, where training data, models weights, architectures and training procedures are publicly available.</p>

<hr/>

[gptsaregpts]: https://arxiv.org/abs/2303.10130
[figureai]: https://www.figure.ai/
[bengio]: https://youtu.be/k9eM6Uwp3bE?feature=shared&t=145
[scalinglaw]: https://arxiv.org/abs/2010.14701
[ilya]: https://www.youtube.com/watch?v=1yvBqasHLZs&t=521s
[reasoning]: https://openai.com/index/learning-to-reason-with-llms/
[mmmu]: https://paperswithcode.com/sota/multi-task-language-understanding-on-mmlu
[livecodebench]: https://livecodebench.github.io/leaderboard.html
[aime2024]: https://www.kaggle.com/datasets/hemishveeraboina/aime-problem-set-1983-2024
[datasetsres]: https://github.com/openai/simple-evals?tab=readme-ov-file#benchmark-results
[statista]: https://www.statista.com/outlook/tmo/artificial-intelligence/worldwide
[hg]: https://huggingface.co/