---
title: "What Happens When AI Meets CFD?"
categories:
  - Blog
tags:
  - Artificial Intelligence
  - Processes
  - Computational Fluid Dynamics (CFD)
authors:
  - pietroscienza
  - me
---

<div class="notice--primary">
Authors: Pietro Scienza, Simone Brazzo
</div>

<div style="background-color: #fcfcfc; color: #2b2b2b; padding: 1rem; border-left: 4px solid #ccc; margin-bottom: 2rem;">
<p>
As in many fields nowadays, the use of AI related tools (e.g. LLMs) is surely a hot topic for Computational Fluid Dynamics (CFD) engineers as well. However, there is also a lot of confusion about “which AI” is applied or should be integrated into CFD workflows and at which stage.
</p>
<p>
In this article we will take a look, first at some possible use cases of AI-CFD interaction, blending LLMs into daily CFD engineering job, moving, later on, to deeper integrations, with a quick overview of the state of art. Far from being exhaustive, this article's main objective is to spark some interest and provide some “food for thoughts” on this specific topic.
</p>
</div>

Let’s start by clarifying what we call a “CFD workflow”. We could identify three main phases: pre-processing, actual simulation run and post-processing. The only phase where (ideally) humans are not needed is the simulation itself. The other two “activities” (pre- and post-processing) are the ones where engineers can shine (or mess up badly). These two are also the parts that could ideally benefit most from modern LLMs.

Let’s begin with the post-processing case. Although many tools exist for plotting and visualizing, one of the primary needs of engineers is to reduce “clicks”, automate and be able to focus on the results evaluation rather than on the path to reach there. Developing templates, scripts, automations procedures, etc. etc. was always there since the beginning of time. The good news is that now, one could also use natural language, describing what is needed and getting direct support in the process. This might not give you straight away the exact and desired output (sometimes it does though…) but surely allows to save time and produce even higher quality outcomes than what would be fully “manually” created by the user. This is something that currently requires an external LLM tool, but what if CFD vendors/developers integrate it directly into their packages…?

Pre-processing can equally benefit from these models. Similarly to what discussed above for post-processing, being able to “interact” with the tools and “explain” how to set up the model, what to modify, how many simulations to run, … could also be of a great benefit, reducing some of the common human errors (typos, wrong values,...) and potentially save time.

Now, two critical points to keep in mind:
* Prompt engineering is essential: one not only should know what they want, but also should be able to clearly express it. Therefore, time should be spent learning how these models think and how to best communicate with them.
*   “Garbage-in/Garbage-out”: LLMs have a lot in common with CFD. They are both based on models with various levels of accuracy and complexity, they both need boundary and initial conditions (prompts, images, data for the LLM; values at domain boundary and inside for CFD) to try to predict what’s next (words for LLMs, quantities evolution in space and time for CFD); and they both produce an output. If both the models are fed with wrong/incomplete/”too coarse” inputs, the results will not be much better in both cases. Therefore, the user should not lower the guard or switch off the brain with either of the two, especially if they are combined.
The path to making these integrations robust might be long, but worth exploring for the potential.

Now, let’s dive at the core level.

AI holds the groundbreaking promise of approximating CFD simulations in less time (days to hours) while obtaining acceptable results.
How is the Machine Learning community facing this challenge? After having reviewed some papers dated from 2020 to 2025 the approaches are various:

* neural networks that take as input the same variables of the simulation and look at the output as target. An interesting example is [this][1], where the authors use a transposed convolutional network to approximate the smoke-visibility model.

* neural networks that are physics-informed, so they take advantage of physics equations that describe fluid dynamics. If you want to experiment with this kind of AI we suggest this [repo][2]. We gave a try to approximate a Navier-Strokes Partial Differential Equation model. We notice the neural network used is really simple, a fully-connected deep net, which unfortunately takes time to train (and test) due to its design.

Anyway, if we look at the open-source ML community (specifically [HuggingFace][hf]) to seek traces of works, datasets and new solutions in the field, unfortunately we don't find much. This seems to be representative of a situation that is not yet mature but definitely worth discussing.

Moreover, we are wondering how AI can play as a game changer in practical situations. We suppose that at least some interesting scenarios can arise from the application of AI to CFD:

* fast prototyping that enables more precise targeting of high-fidelity CFD simulations
* expansion of the global market by allowing CFD simulations to become easier to approach while increasing the demand for more precise methods

[1]: https://www.sciencedirect.com/science/article/abs/pii/S2352710221003867?via%3Dihub
[2]: https://github.com/rezaakb/pinns-torch.git
[hf]: https://huggingface.co

<hr/>

<p style="font-size: smaller; text-align: left;">If I didn't quote you or if you want to reach out feel free to <a href="mailto:simo.brazzo@gmail.com">contact me</a>.</p>
<p style="font-size: smaller; text-align: left;">© [Simone Brazzo] [2025] - Licensed under <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>  with the following additional restriction: this content can be only used to train open-source AI models, where training data, models weights, architectures and training procedures are publicly available.</p>

<hr/>



[v7labs]: https://www.v7labs.com/
[credo]:   https://credo.ai
[wandb]: https://wandb.ai
[ISO42k1]: https://www.iso.org/standard/81230.html