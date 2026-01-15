---
title: "Aidan Smyth - AI Across Borders"
categories:
  - Blog
tags:
  - Artificial Intelligence
  - Society
  - AI Across Borders
layout: single
classes: wide
words_per_minute: 275
---

I’m happy to share the next post in the AI Across Borders series.

This series aims to highlight ideas and experiences in AI from professionals, researchers, engineers, and practitioners around the world, with the goal of fostering cross-border dialogue and connection.

Our guest for this post is [Aidan Smyth][aidan], a Senior AI Engineer at Infineon Technologies. With a background in electronic systems and extensive experience working at the intersection of AI and edge hardware, Aidan brings a valuable industry perspective on deploying intelligent systems in real-world, resource-constrained environments.

We’re excited to share his insights and experience.

Over to you, Aidan!

## Could you introduce yourself and share a bit about your journey into artificial intelligence? What key experiences or turning points led you to your current role as a Senior AI Engineer at Infineon Technologies?

Sure! I graduated in 2018 from Dublin City University and started working in San Jose at
Cypress Semiconductor (which was soon to become Infineon Technologies) around that time.
Today, I am based in California, working out of Infineon’s Irvine office, primarily working
on edge AI-related algorithms on both the hardware and software features.

When I started working as a new college graduate, I joined the IoT MCU division. At the
time, tinyML was beginning to gain attention, and the PSOC team wanted to figure out how
they could leverage edge AI in the current and future PSOC products. My starting point was
developing algorithms for BLE keyfob locationing and tracking, which used machine
learning trained on AoA/AoD/RSSI data from multiple antenna arrays that would be placed
strategically around the car. This data was gathered and simulated, and helped replace
traditional algorithms that didn’t scale well for our customers.

After that work concluded, the focus turned to developing the tooling and algorithms around
the edge AI-focused PSOC Edge MCU. We developed tooling to allow users to bring and
deploy their own TF/TFLite models. I was a key developer in the design of the PSOC Edge’s
voice assistant capabilities (low-power wake word detection, intent and command
recognition, acoustic models to predict speech phonemes, etc.). In recent years I have been
the lead on all things computer vision, which the next version of PSOC Edge MCU will
target design wins in. We have deployed various computer vision algorithms on PSOC Edge,
such as:

* Straightforward single models: Object detection, segmentation, body and hand pose
estimation
* More complicated model systems: Face identification, facial landmark detection, gaze
estimation, optical character recognition
* Future-facing models: Vision Language Models

My current career journey has been made possible through the wonderful mentors and
managers I have had, who encouraged my growth and ambition to take risks, make mistakes,
and go deep when solving problems and creating solutions. What excites me most, looking
forward to the next few years, is the challenge of bringing these incredibly large models to
the true edge (MCU), proving that bigger is not always necessary for useful and
environmentally responsible AI solutions.

## Could you give us more information on PSOC Edge, Infineon’s Edge AI MCU?

A little more info on PSOC Edge. PSOC Edge is Infineon’s MCU that is designed for next
gen responsive compute and control applications, featuring hardware-assisted machine
learning (ML) acceleration.  PSOC Edge provides hardware accelerated neural net compute
support, delivering both &quot;always-on&quot; low power and high-performance operation, in a fully
integrated microcontroller with right-sized peripherals, on-chip memories and state-of-the-art
security. The PSOC Edge devices are based on high performance Arm Cortex-M55,
including Helium DSP support, paired with Arm Ethos-U55 and Cortex-M33 paired
Infineon’s ultra-low power NNLite neural network accelerator for Machine Learning and AI
applications.

## The article on Vision–Language Models I wrote on gradients.zone sparked internal innovation discussions. From your perspective, what makes VLMs particularly compelling for engineers working close to hardware and embedded systems?

VLMs on the edge are incredibly compelling for the following reasons:

* Power savings: Power savings doing this on an edge MCU like PSOC Edge are 1000x
vs using cloud AI engines like ChatGPT etc. When you run on a battery-powered
edge device like many of our customer designs do, this power saving is a game-
changer.
* Responsiveness: If on the edge it becomes truly real-time. No lag or reliance on
internet connectivity. Especially with VLMs where you would need to send large
amount of tokens or even images over the internet.
* Unlocking new human machine interface possibilities: If we bring VLMs to edge we
unlock a host of new HMIs and ways of interacting with these gen AI agents. Your
users may not have access to a keyboard all the time or in a limited capacity, so
having on edge really helps us open it up to other sensors as triggers or inputs.
* Security and privacy: Customers would rather not send their data to some cloud AI. If
we run on PSOC Edge it alleviates such security and privacy concerns around data,
etc.

## As someone operating at the intersection of AI research and industrial constraints, how do you evaluate the gap between cutting-edge multimodal research and what is realistically deployable today?

The gap is narrowing. We have years of experience in taking large models and using
techniques like knowledge distillation, pruning, quantization, etc., to create smaller edge-
friendly models for MCUs. VLMs, while much larger, can still be solved with the same
principles. However, the process is made more difficult with the challenges in data and
compute scale that VLMs require.
I see the main blocker being cost right now. The open-source nature of this work is getting
much better, and groups like HuggingFace and OpenCLIP do an excellent job of trying to
make models as reproducible as possible. However, the engineer who wants to implement
these models for themselves requires now significant investment and backing so they can pay
the GPU providers the costs to run these models. If you care about matching the SOTA
results too, this is usually several hundred GPU hours at least at significant cost.

## In your experience, what organizational or cultural factors most strongly enable meaningful AI innovation inside large, established technology companies?

Meaningful AI innovation is best enabled through backing of a R&amp;D team with funding
and head count, but also with regular IP milestones and deliverables that keep the emphasis
on moving the technology forward and keeping an eye on what technologies are important for
the next wave of innovation. It is important that for a relatively small team, they still find
time to work on enabling future leaning technologies that will help win early market adoption
when the wave arrives.

## Looking ahead, what skills or mental models do you think will distinguish successful AI engineers over the next few years, especially as models become more multimodal and more tightly integrated with physical systems?

The willingness to collaborate and recreate others’ work before improving on it will
distinguish successful engineers over the next few years. Skills in demand right now seem to
be the ability to debug these more complicated VLM systems and their outputs, and the
ability to cook up the perfect training recipe for pre-training or fine-tuning of smaller and
smaller VLM models. I think there will eventually be pushback to making larger and
increasingly power-hungry AI models, and attention will shift to what I am most excited
about: making AI edge-friendly, more sustainable, and accessible to all. Let&#39;s democratize
these big AI models!



<hr/>

<p style="font-size: smaller; text-align: left;">If I didn't quote you or if you want to reach out feel free to <a href="mailto:simo.brazzo@gmail.com">contact me</a>.</p>
<p style="font-size: smaller; text-align: left;">© [Simone Brazzo] - Licensed under <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>  with the following additional restriction: this content can be only used to train open-source AI models, where training data, models weights, architectures and training procedures are publicly available.</p>

[aidan]: https://www.linkedin.com/in/aidan-smyth-02029192/