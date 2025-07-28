---
title: "Multimodal LLMs at the Edge"
categories:
  - Blog
tags:
  - Artificial Intelligence
  - Edge Computing
  - Large Language Models
  - Multimodal AI
layout: single
classes: wide
words_per_minute: 350
---

I always find fashinating to push the limits of AI in terms of trade-off between efficiency and computing resources.

In this example, I explore a shift from the traditional deployment paradigm of multimodal AI models, moving from datacenters to edge devices. By leveraging the browser and GPU, this approach enables the model to run directly on your mobile phone.

How? With these components that I will comment soon:

- [ONNX Runtime Web][onnx-web]
- [transformers.js][transformersjs]
- a model [SmolVLM-256M-Instruct][SmolVLM]

ONNX-RUNTIME is a well established open-source library backed by Microsoft, used for the inference phase of AI/ML model. When you want to abstract your target device from e.g. a specific ML/AI library used to build models, this library is the way.
It has many binding for languages such as Python, C#, JAVA.

Transformers.js is open-source too. It is the JavaScript version of the more famous [transformers][transformerspy] python library.

Finally, the multimodal (image-text-to-text) model SmolVLM-256M-Instruct has been created by the Huggingface team and it is part of a wider family of open-source models. I choose it because enough small to match my needs (just &#x1f605; 256 million parameters). You can find implementation details [here][smolvlm-blog].

Notable features covered in this post:

* usage of [webgpu][webgpu] backend to take advantage of gpu shaders 

So let's jump into the code, with a simple html page that will host a button to run SmolVLM.
You can run the project by [http-serve][http-serve].

[http-serve]: https://www.npmjs.com/package/http-server
[transformersjs]: https://github.com/huggingface/transformers.js
[transformerspy]: https://github.com/huggingface/transformers
[SmolVLM]: https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct
[onnx-web]: https://onnxruntime.ai/docs/get-started/with-javascript/web.html
[smolvlm-blog]: https://huggingface.co/blog/smolvlm
[webgpu]: https://github.com/gpuweb/gpuweb

```html
<!DOCTYPE html>
<html>
<head>
  <title>SmolVLM Demo</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/onnxruntime-web/1.20.1/ort.webgpu.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjs/14.0.1/math.js" integrity="sha512-ldafwBWmh8q0wplbjDzai4As66n/6e0kxw51a+LRJ6+aZ27t0oGpz7HH5dUh+MwWLacrsF8cGT4zR0p2S3QHtA==" crossorigin="anonymous" referrerpolicy="no-referrer"></script>
  <script type="module" src="smolvlm.js"></script>
</head>
<body>
    <h1>SmolVLM Image Captioning</h1>
    <button id="runButton">Run Model</button>
    <div id="result"></div>

    <script type="module">
    // Import the function from your module
    import { runSmolVLM } from './smolvlm.js';
    
    // Add event listener to button
    document.getElementById('runButton').addEventListener('click', async () => {
        try {
        await runSmolVLM();
        } catch (error) {
        console.error("Error running SmolVLM:", error);
        document.getElementById('result').textContent = "Error: " + error.message;
        }
    });
    </script>
</body>
</html>
```

And now, the core part. This js file contains the following parts:

* loading configuration of SmolVLM by using Huggingface utilities.
* load the 3 quantized models (4-bit quantization): vision encoder, embedding table and the decoder part (LLM). You can find the model files [here][onnxmodels].
* perform the preprocessing with tokenization of image and text.
* perform the inference with the autoregressive loop to generate text.

I used to add some comments with shapes for ease of dubugging. This can help in case you want to reproduce the code.

[onnxmodels]: https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct/tree/main/onnx

```javascript
import { 
  AutoProcessor,
  load_image,
  AutoConfig
} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.3.3';

class SmolVLMInference {
    constructor(config) {
      // Model configuration
      this.modelId = "HuggingFaceTB/SmolVLM-256M-Instruct";
      this.config = {
        text_config: {
          num_key_value_heads: config.text_config.num_key_value_heads,
          head_dim: config.text_config.head_dim,
          num_hidden_layers: config.text_config.num_hidden_layers,
          eos_token_id: config.text_config.eos_token_id
        },
        image_token_id: config.image_token_id
      };
      
      // Initialize sessions and processor
      this.visionSession = null;
      this.embedSession = null;
      this.decoderSession = null;
      this.processor = null;
      
      // Model parameters from config
      this.numKeyValueHeads = this.config.text_config.num_key_value_heads;
      this.headDim = this.config.text_config.head_dim;
      this.numHiddenLayers = this.config.text_config.num_hidden_layers;
      this.eosTokenId = this.config.text_config.eos_token_id;
      this.imageTokenId = this.config.image_token_id;
    }
  
    // Initialize ONNX sessions
    async loadModels() {
      try {
        console.log("Loading ONNX models...");
        
        // Load all three models in parallel
        [this.visionSession, this.embedSession, this.decoderSession] = await Promise.all([
          ort.InferenceSession.create('./vision_encoder_q4.onnx', { executionProviders: ['webgpu'] }),
          ort.InferenceSession.create('./embed_tokens_q4.onnx', { executionProviders: ['webgpu'] }),
          ort.InferenceSession.create('./decoder_model_merged_q4.onnx', { executionProviders: ['webgpu'] })
        ]);
        
        console.log("Models loaded successfully!");
        return true;
      } catch (error) {
        console.error("Error loading models:", error);
        return false;
      }
    }

    async officialPreproc(imageUrl, question){

      const image1 = await load_image(imageUrl);

      // Load processor and model
      const model_id = "HuggingFaceTB/SmolVLM-256M-Instruct";
      this.processor = await AutoProcessor.from_pretrained(model_id);

      const messages = [
          {
              role: "user",
              content: [
                  { type: "image" },
                  { type: "text", text: question },
              ],
          },
      ];
      const prompt = this.processor.apply_chat_template(messages, { tokenize: false, add_generation_prompt: true });
      const inputs = await this.processor(prompt, [image1]);

      return inputs;
    }
  
    // Main inference function
    async generateText(imageUrl, question, maxNewTokens = 1024) {
      try {

        const officialInputProcessing = await this.officialPreproc(imageUrl, question);
        
        // Prepare decoder inputs
        const batchSize = 1;
        let pastKeyValues = {};
        for (let layer = 0; layer < this.numHiddenLayers; layer++) {
          for (let kv of ['key', 'value']) {
            pastKeyValues[`past_key_values.${layer}.${kv}`] = new ort.Tensor(
              'float32', 
              new Float32Array(0), 
              [batchSize, this.numKeyValueHeads, 0, this.headDim]
            );
          }
        }
        
        let imageFeatures = null;
        let inputIds = officialInputProcessing.input_ids;
        let attentionMask = officialInputProcessing.attention_mask;
        
        // Calculate position IDs
        let positionIds = this.calculatePositionIds(attentionMask);
        
        // Generation loop
        let generatedTokens = [];
        let outputText = "";
        
        console.log("Starting generation...");
        
        for (let i = 0; i < maxNewTokens; i++) {

          // Get token embeddings
          const inputIdsArray = Array.from(this.getTensorData(inputIds));
          const embedFeed = { 'input_ids': inputIds };
          const embedResult = await this.embedSession.run(embedFeed);

          // [1, 876, 576]
          let inputsEmbeds = embedResult.inputs_embeds;
          
          // Process image if needed
          if (imageFeatures === null) {

            const imageTokenCount = inputIdsArray.filter(num => num === BigInt(this.imageTokenId)).length;

            const visionFeed = {
              'pixel_values': officialInputProcessing.pixel_values,
              'pixel_attention_mask': officialInputProcessing.pixel_attention_mask
            };
            
            const visionResult = await this.visionSession.run(visionFeed);

            // imageFeatures.shape = [13, 64, 576]
            const firstDim = visionResult.image_features.dims[0] * visionResult.image_features.dims[1];
            const secDim = visionResult.image_features.dims[2];
            // [13, 64, 576] -> [479232] contiguous
            imageFeatures = Array.from(this.getTensorData(visionResult.image_features));
            // [13, 64, 576] -> [832, 576]
            imageFeatures = math.reshape(imageFeatures, [firstDim, secDim]);

            // there must be image_token * firstDim tokens in inputsEmbeds, then replace each position (second dim) with the index from imageFeatures

            if (imageTokenCount != firstDim) {
              return "Error, invalid number of image tokens";
            }

            const origDims = inputsEmbeds.dims; // [1, 876, 576]
            const origLocation = inputsEmbeds.location; // cpu
            const origType = inputsEmbeds.type; // float32
            const origSize = inputsEmbeds.size; // 504576

            // [504576] contiguous
            let inputsEmbedsArray = Array.from(this.getTensorData(inputsEmbeds)); 
            // [504576] -> [876, 576]
            inputsEmbedsArray = math.reshape(inputsEmbedsArray, [inputsEmbeds.dims[1], inputsEmbeds.dims[2]]); // first dimension [1] is not effective here

            // replace with imageFeatures
            let imgFeaturesCnt = 0;
            for (let i = 0; i < inputIdsArray.length; i++){
              if (inputIdsArray[i] == BigInt(this.imageTokenId)) {
                inputsEmbedsArray[i] = imageFeatures[imgFeaturesCnt];
                imgFeaturesCnt += 1;
              }
            }

            // [876, 576] -> [504576]
            inputsEmbedsArray = math.reshape(inputsEmbedsArray, [inputsEmbeds.size]);

            // convert the array back to tensor (cpu)
            inputsEmbeds = new ort.Tensor("float32", new Float32Array(inputsEmbedsArray), [inputsEmbeds.size]);
            inputsEmbeds = inputsEmbeds.reshape(origDims);

            if (origDims !== inputsEmbeds.dims || origLocation !== inputsEmbeds.location || origType !== inputsEmbeds.type || origSize !== inputsEmbeds.size) {
              return "Error, convertion of inputsEmbed failed";
            }

          }
          
          // Run decoder model
          const decoderFeeds = {
            'inputs_embeds': inputsEmbeds,
            'attention_mask': attentionMask,
            'position_ids': positionIds,
            ...pastKeyValues  // [1, 3, 0 ,64]
          };
          
          const decoderResults = await this.decoderSession.run(decoderFeeds);
          
          // [1, 876, 49280]
          const logits = decoderResults.logits; 

          // we take the entire object, remove the logits with effect on [decoderResults]
          const presentKeyValues = decoderResults;
          delete presentKeyValues.logits;
          
          // Get next token (argmax of last logits)
          const nextToken = this.getNextToken(logits);
          
          // Update for next iteration
          inputIds = new ort.Tensor('int64', new BigInt64Array([BigInt(nextToken)]), [1, 1]);
          attentionMask = new ort.Tensor('int64', new BigInt64Array([1n]), [1, 1]);
          positionIds = new ort.Tensor('int64', new BigInt64Array([BigInt(this.getTensorData(positionIds).at(-1) + BigInt(1))]), [1, 1]);
          
          // Update past key values
          this.updatePastKV(presentKeyValues, pastKeyValues);
          
          // Add token to generated sequence
          generatedTokens.push(nextToken);
          
          // Decode token and add to output text
          const tokenText = this.processor.decode([nextToken]);
          outputText += tokenText;
          
          // Optional streaming output
          if (i % 5 === 0) {
            console.log("Generation progress:", outputText);
          }
          
          // Check for EOS token
          if (nextToken === this.eosTokenId) {
            break;
          }
        }
        
        console.log("Generation complete!");
        return outputText;
      } catch (error) {
        console.error("Error in generation:", error);
        return "An error occurred during text generation.";
      }
    }

    // update KVs
    updatePastKV(presentKV, pastKV) {

      console.log("updatePastKV");

      for (let layer = 0; layer < this.numHiddenLayers; layer++) {
        for (let kv of ['key', 'value']) {
          pastKV[`past_key_values.${layer}.${kv}`] = presentKV[`present.${layer}.${kv}`];
        }
      }
    }
  
    // Helper to calculate position IDs from attention mask
    calculatePositionIds(attentionMask) {
      const attentionArray = this.getTensorData(attentionMask);
      const positionArray = new BigInt64Array(attentionArray.length);
      
      let position = 0n;
      for (let i = 0; i < attentionArray.length; i++) {
        if (attentionArray[i] === 1n) {
          positionArray[i] = BigInt(position);
          position++;
        } else {
          positionArray[i] = 0n;
        }
      }
      
      return new ort.Tensor('int64', positionArray, attentionMask.dims);
    }
  
    // Helper to get next token from logits
    getNextToken(logits) {
      // Get the last token's logits
      const lastLogits = Array.from(this.getTensorData(logits).slice(-logits.dims[2]));
      
      // Find the index of the maximum value (argmax)
      let maxIndex = 0;
      let maxValue = lastLogits[0];
      
      for (let i = 1; i < lastLogits.length; i++) {
        if (lastLogits[i] > maxValue) {
          maxValue = lastLogits[i];
          maxIndex = i;
        }
      }
      
      return maxIndex;
    }
  
    // Helper to get tensor data as array
    getTensorData(tensor) {
      return tensor.data;
    }
  }
  
  // Usage example
  async function runSmolVLM() {

    let model_id = "HuggingFaceTB/SmolVLM-256M-Instruct";
    const config = await AutoConfig.from_pretrained(model_id);
    const inferenceEngine = new SmolVLMInference(config);
    
    // Step 1: Load models
    const modelsLoaded = await inferenceEngine.loadModels();
    if (!modelsLoaded) {
      console.error("Failed to load models");
      return;
    }
    
    // Step 2: Run inference
    const imageUrl = "./Statue-of-Liberty-Island-New-York-Bay.jpg";
    const question = "Can you describe this image?";
    
    console.log("Running inference on image:", imageUrl);
    console.log("Question:", question);
    
    const result = await inferenceEngine.generateText(imageUrl, question);
    
    // Step 3: Show results
    console.log("Generated text:");
    console.log(result);
    
    // Display in UI if needed
    if (document.getElementById('result')) {
      document.getElementById('result').textContent = result;
    }
  }

// Add this at the bottom of your smolvlm.js file
export { SmolVLMInference, runSmolVLM };
```

And this is the result for this beautiful image and prompt:

**Q**: "Can you describe this image?"

![Input image]({{ site.baseurl }}/assets/images/Statue-of-Liberty-Island-New-York-Bay.jpg)


**A-Q4**:
> In the foreground of the picture we can see a statue of the statue of the person holding the crown. In the background we can see the sky, the sky is blue in the background we can see the sky, the sky is clear. On the right side we can see the buildings, the buildings are in white and brown color.

I hoped for better &#x1f635;! Redundancy and repetition are clear problems here.

What about the model output by using higher precision (e.g. fp16?):

**A-FP16**:
> The image depicts a large, historic statue of Liberty situated on a small island in a body of water. The statue is a green, cylindrical structure with a human figure atop > it. The figure is depicted as a woman wearing a flowing gown, with her hair tied back. The statue is mounted on a pedestal that is supported by a cylindrical tower. The > > pedestal is rectangular and appears to be made of stone or concrete.
The water surrounding the island is calm, with gentle ripples indicating the gentle movement of the water. The water is a deep blue color, reflecting the sky above. In the background, there are several tall buildings, including a modern skyscraper and a more traditional building. These buildings are constructed with glass and steel, and they are positioned in a grid-like pattern.
The sky above is clear, with a few clouds visible, indicating fair weather. The overall atmosphere of the image is calm and serene, with the statue and the surrounding environment providing a sense of peace and tranquility.
To summarize:
- **Statue**: The large green statue of Liberty is located on a small island.
- **Pedestal**: The pedestal is rectangular and supports the statue.
- **Water**: The water is calm, with gentle ripples indicating the movement of the water.
- **Buildings**: The buildings are tall and constructed with glass and steel, with a grid-like pattern.
- **Sky**: The sky is clear, with a few clouds visible.
This image captures a moment of tranquility and historical significance, highlighting the statue's role as a symbol of freedom and liberty. The calm water and the clear sky suggest a peaceful environment, which is ideal for public art and reflection.

Better at the price of increased runtime and space!

Note: pay attention that fp16 is poorly supported by mobile device browsers.

<hr/>

<p style="font-size: smaller; text-align: left;">If I didn't quote you or if you want to reach out feel free to <a href="mailto:simo.brazzo@gmail.com">contact me</a>.</p>
<p style="font-size: smaller; text-align: left;">© [Simone Brazzo] [2025] - Licensed under <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>  with the following additional restriction: this content can be only used to train open-source AI models, where training data, models weights, architectures and training procedures are publicly available.</p>

<hr/>