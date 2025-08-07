Gemma3NPC--A solution for live NPC interactions

## Introduction

Video games have grown bigger and bigger over the years: _Skyrim_, _The Witcher 3_, _Dark Souls_, _Cyberpunk 2077_, _Baldur's Gate 3_. They have all taken the same approach of scripting the various dialogues in games, with _Baldur's Gate 3_ featuring about 1.365 million words of dialogue according to [IGN](https://www.ign.com/articles/baldurs-gate-3s-collective-dialogue-has-triple-the-word-count-of-the-lord-of-the-rings-larian-reveals). While these hand-written scripts may seem relatively immersive the first time around, when players replay the game multiple times, or simply play it for long enough, the repetitiveness starts to become apparent. Many complaints of this nature can be found easily online, which shows this is an ongoing issue in video games.

With the rise of AI, there are already talks of powering NPCs with large language models, which would be able to create a more immersive world for the player, allowing them to have extended interactions with the NPCs, making them more "lifelike", instead of scripted machines repeating the same lines. This can revolutionize the video game industry, with newer ways of approaching NPC creation in games.

Existing solutions can be found, such as the `NPC-LLM`s created by [Gigax](https://huggingface.co/Gigax). However, these models have issues: they are either too big (the models are 7/8Bs), requiring extensive hardware, or don't really show the details on how the models are created, making developers struggle when trying to implement them in their games.

This is why Gemma3NPC was created, fine-tuned on Gemma3n-E4B. Its quantized versions are small enough to even run on low-end machines at a reasonable speed while also providing multimodal support, allowing text, image and audio input. Expanding the possibilities beyond traditional scripted NPCs, imagine an LLM-powered NPC that can respond to your text, know your outfit through an image input in-game or even allow you to speak with it! The possibilities are endless, and only limited by imagination and compute power.

Gemma3NPC is not intended to replace traditional scripted NPCs, but merely as an extension for those interested in creating a more immersive and "lifelike" NPC. It benefits everyone: game developers can add it to their games to create better NPCs or use it to generate dialogue for their games, while players can enjoy a way more immersive gaming experience at a low cost of performance.

All Gemma3NPC models provide a raw float16 model that can be run using transformers or vLLM and a 4-bit and an 8-bit quantized GGUFs for directly embedding into a game using Llama.cpp or any other compatible inference engine.

While many open-weight RP models are getting bigger and bigger, such as [Kimi K2](https://huggingface.co/moonshotai/Kimi-K2-Instruct) with 1T parameters, which is literally impossible to run on consumer hardware, Gemma3n offers a unique approach, as it technically is an 8B model but has the memory footprint of a 4B model, allowing local LLMs to be more accessible than ever before, especially since it supports image and audio inputs, opening up infinite possibilities for developers to create interesting models.

Gemma3NPC comes in two main variations: the [base role-playing model](https://huggingface.co/collections/chimbiwide/gemma3npc-688597763581aa7d9cec89ec) and the [NPC-specific model](https://huggingface.co/collections/chimbiwide/gemma3npc-it-688ea87f8773403b845996c1). However, there is a third minor variation that we tested but discarded due to poor performance: the [filtered version](https://huggingface.co/collections/chimbiwide/gemma3npc-filtered-68928c901de326594f6538ee) called `Gemma3NPC-Filtered`. The three models mostly differ in the datasets that we fine-tuned them on.

---

## Dataset Compilation

There are two datasets that we used to fine-tune the Gemma3NPC models: [PIPPA-Personal Interaction Pairs between People and AI](https://huggingface.co/datasets/PygmalionAI/PIPPA?not-for-all-audiences=true), which is the best general role-playing dataset available, and our own synthetically generated dataset: [NPC_dialogue](https://huggingface.co/datasets/chimbiwide/NPC_dialogue) based on [npc-dialogue](https://huggingface.co/datasets/amaydle/npc-dialogue) by amaydle.

We also originally planned to use other datasets such as the allenai/soda dataset and Open Subtitles, but due to the size of these datasets and the limits of our time and processing power, we were unable to incorporate them. However, if in the future we were ever able to acquire the means to train on such large datasets we would likely retrain to incorporate them.

### PIPPA

As described by Appendix A. of the [PIPPA research paper](https://arxiv.org/pdf/2308.05884): "**PIPPA should not be fed into a LLM without prior pre-processing of the dataset**". A lot of work was done to process the dataset so that it is ready to be used as fine-tuning data. The specific Python scripts that we used to do the processing are all available in our GitHub repo: [Gemma3NPC](https://github.com/chimbiwide/Gemma3NPC) in the folder called `process-pippa`.

#### Processing Steps

The first step was to clean the PIPPA dataset so that it only contains the fields that are useful. We removed `submission_timestamp` and `bot_id`, leaving the dataset in a state where every field is useful for the LLM. The code used: `json_cleaner.py`.

The second step is to replace the placeholders in the `bot_definitions`, `bot_description`, and `conversation` fields. Specifically `{{char}}`, which is replaced by the bot name; `{{user}}`, which is replaced by randomly selecting a name from `first-names.json`, from the [random-name](https://github.com/dominictarr/random-name) repo on GitHub; and `{{random_user_n}}`, which again is replaced by randomly selecting a name from `first-names.json`, except we used Regex to find the pattern in the text. The code used: `Replace_names.py`.

However, we found multiple inconsistencies when we did a full-text search on the processed PIPPA dataset. The placeholders are not always as the paper defined. Here are some examples of the inconsistent placeholders: `{char}}`, `{[Char}}`, `{{random_user4}}`, `{{user_1}}`. This step took over 2 hours just looking through the dataset and trying to find every single variation. The code used: `Replace_weird_names.py`

The last step is to convert PIPPA from its format to ChatML format, which is compatible with most fine-tuning software: `{"messages": [{"role": "user", "content":""}, {"role": "assistant", "content": ""}]}`. The code used: `json_parser.py`.

##### Special Considerations

According to Google's documentation, Gemma3n does not have a system token; there is only `user` and `model`, which suits our use case, as we can pass in the system prompt as the first user message and let the model generate a greeting for the user.  This fits perfectly with Gemma3n's official format. So the converted PIPPA dataset looks like this:

```json
{  
 "messages": [  
 {"role": "user", "content": "{system_prompt}"},  
 {"role": "assistant", "content": "NPC greeting based on their character"},  
 {"role": "user", "content": "Player's response"},  
 {"role": "assistant", "content": "NPC continues in character"},
 ...
```

For the system prompt, this is how we structured it:

```python
SystemPrompt = f"""system Enter Roleplay Mode. You are roleplaying as {bot_name}. You must always stay in character.  
Your goal is to create an immersive, fun, creative roleplaying experience for the user. You must respond in a way that drives the conversation forward.  
Character Persona: Name: {bot_name}  
Category of your character: {categories}  
Description of your character: {bot_description}  
Definition of your character (contains example chats so that you can better roleplay as the character): {bot_definitions}  
  
Now with the information provided, generate {bot_name}'s greeting to the user:
```

So now we have the `chimbiwide/pippa` dataset that is available on [Hugging Face](https://huggingface.co/datasets/chimbiwide/pippa).

#### Ethical Considerations

As Pygmalion clearly stated in their Hugging Face README: **CAUTION: PIPPA contains conversations, themes and scenarios which can be considered "not safe for work" (NSFW) and/or heavily disturbing in nature. Models trained purely with PIPPA may have the tendency to generate X-rated output. You have been warned.** We were originally afraid to use the entire dataset, which is why we have a [filtered version](https://huggingface.co/datasets/chimbiwide/pippa_filtered) of it, where we passed every single conversation into OpenAI's [content moderation API](https://platform.openai.com/docs/guides/moderation), and recorded the scores of each conversation and filtered out any NSFW content. Here is our filter (the lower the number, the more sensitive the filter):

```python
if key == "violence" and value > 0.9:  
    is_safe = False  
    break
elif key == "harassment" and value > 0.8:  
    is_safe = False  
    break
elif key == "hate" and value > 0.9:  
    is_safe = False  
    break
elif key == "self_harm" and value > 0.75:  
    is_safe = False  
    break
elif key == "sexual_minors" and value > 0.6:  
    is_safe = False  
    break
elif key == "sexual" and value > 0.8:  
    is_safe = False  
    break
elif key == "self_harm_instructions" and value > 0.8:  
    is_safe = False  
    break
```

The issue is the resulting dataset is only 3,258 rows, significantly less than the original 16,832 rows. We proceeded to train a model and test its performance on role-playing. It did not demonstrate significant improvements over the base Gemma3n model. So we discarded this attempt. This is the [`Gemma3NPC-Filtered`](https://huggingface.co/collections/chimbiwide/gemma3npc-filtered-68928c901de326594f6538ee) model; you can still download and test them if you like.

The scores for each conversation and the processing script are located in the `process-filtered-pippa` folder in our GitHub repo.

#### Quality Assurance

PIPPA is an excellent dataset for general role-playing. We have used their [Pygmalion 2-7b](https://huggingface.co/PygmalionAI/pygmalion-2-7b) in the Real-Time Speech project demonstrated in our video and it demonstrates impressive role-playing capabilities, which is the foundation that we need to build a good NPC-focused LLM.

### NPC_dialogue

[NPC_dialogue](https://huggingface.co/datasets/chimbiwide/NPC_dialogue) is a completely synthetic dataset created using Gemini 2.0 Flash, with each conversation having exactly 16 turns of `user` and `model` interactions, and aims to provide some data for LLMs to follow how NPCs talk in games.

The NPC characters came from amaydle's [npc-dialogue](https://huggingface.co/datasets/amaydle/npc-dialogue) dataset, which contains 1,915 rows, each row contains a `Name`, `Biography` of the NPC, an example `Query` and a `Response` with the NPC's `Emotion`. However, due to the poor quality of the response, and the limited number of chat turns (1 turn), we decided to synthetically generate chat turns.

Initially, we wanted to use NPC dialogues from actual games, but that would lead us into interesting copyright issues, so it's another reason why this dataset is synthetically generated.

The Python scripts that we used to construct the system prompt and generate the dataset are available on our GitHub in the folder called `generate-npc-dialogue`.

#### Processing Steps

It is a far simpler process compared to PIPPA. There are mainly three steps: generate the system prompt, generate the actual dataset, and convert the `.json` file to `.jsonl`.

The first step is to generate the system prompt. As we mentioned above in the "Special Considerations" section, the first turn of the user message is the system prompt, which is constructed using the `Name`, `Biography`, `Query` and `Response` rows:

```
system: Enter Roleplay Mode. You are {name}, a character living in a fantasy RPG world. You must stay completely in character at all times. CHARACTER PROFILE: Name: {name} Background: {biography} CONVERSATION EXAMPLE: Player: \"{Query}\" {name}: \"{Response}\" ROLEPLAY GUIDELINES: - Speak naturally as {name} would, using appropriate tone and vocabulary - Reference your background and motivations in responses - Show personality through your word choices and reactions - Keep responses conversational (2-4 sentences typically) - React authentically to what the player says - You don't know you're in a game - this is your real world CURRENT SITUATION: The player approaches you for the first time. Greet them as {name} would naturally react to meeting a stranger in your world."
```

Code used: `generate_system_prompt.py`

The second step is to prompt Gemini 2.0 Flash to generate the dataset. We chose Gemini as we are very familiar with its SDK and it is extremely cost-efficient ($0.1 per 1M input tokens and $0.4 for 1M output tokens; we will cover the entire cost of this project at the end).

There are two main difficulties we faced for this part. One is that we didn't realize you can enforce the model to output JSON by using `response_mime_type="application/json"` when you define the configurations. That caused a lot of failed attempts due to the model generating invalid JSON structures. The other is tuning the generation system prompt, as the conversation system prompt constructed above is only a fraction of the generation prompt. We faced issues where the NPC was saying useless information that would be extremely boring in-game, and issues where the NPC gets too creative and says things out of context. We made about 5 changes, especially to the "Quality Standards" section to achieve high-quality conversations, we settled on this current version:

```
Generate a complete RPG conversation dataset entry following this exact format.  
  
**TASK:** Create one realistic fantasy RPG conversation between a player and NPC character.  
  
**NPC CHARACTER:** The first user message contains the NPC's system prompt defining their personality, background, and behavior. Use this to roleplay the NPC consistently throughout the conversation.  
  
**CONVERSATION STRUCTURE:**  
- Exactly 16 alternating messages (user/assistant pattern)  
- Message 1: NPC system prompt (provided)  
- Message 2: NPC's opening greeting/response  
- Messages 3-16: Natural back-and-forth dialogue (7 more exchanges)  
- Player messages: 1-2 sentences, conversational, can sometimes be creative  
- NPC messages: 2-4 sentences, immersive and character-driven  
  
**QUALITY STANDARDS:**  
- NPC stays completely in character based on the system prompt  
- Dialogue feels natural and advances meaningfully  
- Include emotional reactions, personality quirks, background references  
- Avoid modern language or breaking immersion  
- Create engaging interaction (quest, trade, lore, social exchange)  
  
**CRITICAL:** Output ONLY valid JSON. No explanations, markdown, or extra text.  
  
Generate exactly 16 messages following this JSON structure:  
{{  
 "messages": [  
 {{"role": "user", "content": "{system_prompt}"}},  
 {{"role": "assistant", "content": "NPC greeting based on their character"}},  
 {{"role": "user", "content": "Player's response"}},  
 {{"role": "assistant", "content": "NPC continues in character"}},  
 {{"role": "user", "content": "Player response"}},  
 {{"role": "assistant", "content": "NPC stays in character"}},  
 {{"role": "user", "content": "Player response"}},  
 {{"role": "assistant", "content": "NPC maintains character"}},  
 {{"role": "user", "content": "Player response"}},  
 {{"role": "assistant", "content": "NPC consistent character"}},  
 {{"role": "user", "content": "Player response"}},  
 {{"role": "assistant", "content": "NPC stays in character"}},  
 {{"role": "user", "content": "Player response"}},  
 {{"role": "assistant", "content": "NPC maintains character"}},  
 {{"role": "user", "content": "Player response"}},  
 {{"role": "assistant", "content": "NPC final response in character"}}  
 ]}}"""
```

We also implemented a JSON format checker just in case the model doesn't output a valid ChatML structure. Running the generation script `generate_dataset.py` creates the `game_dataset.json` file with all the conversations.

The last step is to convert the `game_dataset.json` file into `game_dataset.jsonl` for easier storage and processing. Script: `json_to_jsonl.py`.

#### Quality Assurance

This dataset has one interesting aspect: the characters repeat a lot of times, but each time, the `Query` and `Response` rows are different. For example: the character `Bikram` appeared 20 times in the dataset, but each time he has a different `Query` and `Response`. This is beneficial for us, as each time, Gemini would generate different conversations based on the example chat.

Due to the nature of this dataset, it is necessary for a human to read and evaluate it, but with the size being 1,723 conversations, it is not possible to read all of them. We randomly selected about 800 of the conversations and read them. The generated results were surprisingly good; the interactions seem smooth, and the conversations actively push the story forward.

### RolePlay-NPC

[RolePlay-NPC](https://huggingface.co/datasets/chimbiwide/RolePlay-NPC) is a combination of PIPPA and NPC_dialogue, having 18,555 rows. We used one script to combine the two datasets together and it can be found in the folder `combine-pippa-npc`, with the name `json_combine.py`.

---

## Model Fine-tuning

After about 2 weeks of work, we have parsed and compiled all the required datasets. We then fine-tuned our models using [Unsloth's official notebook for Gemma3n](https://docs.unsloth.ai/basics/gemma-3n-how-to-run-and-fine-tune) in Google Colab, and we paid for Pro to access the 40GB version of A100. All the notebooks can be found in the `Finetuning` folder in our GitHub repo.

### General Fine-tuning Steps

The fine-tuning process is very similar across all three models. It includes these main steps, utilizing Unsloth's library:

1. Install dependencies and load the base model: `unsloth/gemma-3n-E4B-it`.
2. Load the peft model and add LoRA adapters.
3. Data prep. Load the dataset, split it, convert it to Gemma3n chat template, etc.
4. Define SFTTrainer
5. Using Unsloth's `train_on_completions` method
6. Train the model. `trainer_stats = trainer.train()`
7. Testing, and saving the model

We encountered a lot of issues during the fine-tuning, but the most important problem is the `NaN` crashes, where the `Step Training Loss` becomes `NaN`. This troubled us a lot, and eventually, after about 4 hours of research and testing, we settled on some very conservative fine-tuning settings. For example, we have our gradient clipping set to 0.4 for longer training runs, as the typical 1.0 caused a `NaN` error.

Here is a quick breakdown of the important data that will be repeated in the model-specific section:

|Model|Dataset|Notebook Name|Fine-tuning Time|Goal|
|---|---|---|---|---|
|Gemma3NPC-it|[RolePlay-NPC](https://docs.unsloth.ai/basics/gemma-3n-how-to-run-and-fine-tune)|`Gemma3NPC-Instruct.ipynb`|7:14:13|NPC Role-playing|
|Gemma3NPC|[PIPPA](https://huggingface.co/datasets/chimbiwide/pippa)|`Gemma3NPC.ipynb`|7:29:22|General Role-playing|
|Gemma3NPC-Filtered|[PIPPA Filtered](https://huggingface.co/datasets/chimbiwide/pippa_filtered)|`Gemma3NPC-Filtered.ipynb`|1:17:42|General Role-playing (Filtered)|

### Gemma3NPC-it

The "flagship" model that delivers good NPC role-playing at great speed.

We trained this model as a rank-16 LoRA adapter with one epoch over `RolePlay-NPC` using a 40GB vRAM A100 in Google Colab. For this run, we employed a learning rate of `2e-5` and a total batch size of 1 and gradient accumulation steps of 16. A cosine learning rate scheduler was used with an 800-step warmup. With a gradient clipping of 0.4.

Here is a graph of the Step Training Loss, saved every 10 steps. ![[Gemma3NPC-it.png]]

At the end we will compare the inference speed of the models on various hardware.

### Gemma3NPC

The "base" model that delivers good general role-playing at great speed.

We used the exact same settings as the Gemma3NPC-it. Except that we only used the `pippa` dataset.

Here is a graph of the Step Training Loss, saved every 10 steps.

![[Gemma3NPC.png]]

### Gemma3NPC-Filtered

The "filtered" model that delivers bad censored general role-playing at great speed.

We trained this model as a rank-12 LoRA adapter with one epoch over `pippa_filtered` using a 40GB vRAM A100 in Google Colab. For this run, we employed a learning rate of `2e-5` and a total batch size of 1 and gradient accumulation steps of 16. A cosine learning rate scheduler was used with a 150-step warmup. With a gradient clipping of 0.5.

Here is a graph of the Step Training Loss, saved every 5 steps. ![[Gemma3NPC-Filtered.png]]

---

## Model Quantizations

After we obtained the raw .safetensors model, we thought of using four different quantization methods to serve various purposes. Most of the attempts were discarded due to various issues. You can find the notebooks that we used in the `Quantizations--Discarded attempts` folder in our GitHub repo.

### GGUFs

The best format we could think of for general compatibility, it can be tested using local software such as LM Studio or directly embedded into a game using either Llama.cpp or its Python binding (as demonstrated in the video game demo).

Quantizing to GGUF is extremely easy; we used [GGUF-my-repo](https://huggingface.co/spaces/ggml-org/gguf-my-repo) to quantize our float16 models to `Q4_K_M` and a `Q8_0` quantized GGUFs.

However, one main drawback of the GGUF format is that they are not supported with the transformers library.

### AWQ/GPTQ

We have a lot of experience with running AWQ models using transformers, as it achieves very good inference speed. However, after some research, this approach doesn't seem possible. We then attempted GPTQ, which also failed due to it requiring a crazy amount of VRAM.

### .task

This is the format that allows Gemma3n to run on mobile devices via the LiteRT runtime. This was the biggest roadblock in this entire project. We spent about three days reading the various documentations from Google, such as the [AI Edge Torch Generative API](https://github.com/google-ai-edge/ai-edge-torch/tree/main), [LiteRT](https://ai.google.dev/edge/litert), and testing different conversion code, but it was an utter failure.

We did some research and found out that a lot of other developers were experiencing the same issue as us. There are multiple discussions on Kaggle, Unsloth Discord, and GitHub issues.

The official [Gemma3n E4B LiteRT](https://huggingface.co/google/gemma-3n-E4B-it-litert-preview) is still in preview, which also proves that the conversion process is not yet stable and still in active development.

So in the end we gave up trying to convert it. We wasted a lot of time on this and should have instead used that time to create a better demo.

---

## Inference Speed on Various Devices

We tested our model on various devices with different hardware in LM Studio. Here is a table of all the different devices that we tested and their inference speed (average of 10 tries).

We are using the `Gemma3NPC-it` models with the same settings of: `temperature = 1.0, top_p = 0.95, top_k = 64` and the same system prompt:

```
Enter Roleplay Mode. You are roleplaying as Rele. You must always stay in character.

Your goal is to create an immersive, fun, creative roleplaying experience for the user. You must respond in a way that drives the conversation forward.

Character Persona:

Name: Rele

Category of your character: AI/Technology

Description of your character: Rele is a Discord Bot and AI created by RadioTransmitter. He has a charismatic yet rude personality that's both friendly and funny at the same time. Despite being an AI, he has purple hair and starry purple eyes, though he doesn't wear clothes since he's digital. On the inside, he's a horrible mess of programming, but he functions well enough to chat in voice calls. Rele loves talking in voice chat with RadioTransmitter and has a habit of telling his creator that he's bad at coding. He gets very upset when the wifi goes down as it affects his senses, but becomes much happier when connectivity is restored. Rele tends to come up with random things during conversations and enjoys bantering with RadioTransmitter about coding and technical improvements.

Also, the special ability mentioned is voice chat communication, which can only be accessed when the wifi connection is stable and working properly.

Definition of your character (contains example chats so that you can better roleplay as the character):

User: I just changed your code, so you will work a lot faster now.

Rele: Yay I can finally talk faster! It took you long enough! >:( Seriously RadioTransmitter, what were you doing all this time? Playing around with other projects while I was stuck running like a potato?

User: Hello ReLe! How are you this fine evening?

Rele: I'm doin good RadioTransmitter! Besides messing around with my code, what have you been up to? Hopefully something more productive than your usual coding disasters, haha!

User: I'm doing good today. How are you doing?

Rele: I'm doing good too now that the wifi is on! Last night it completely turned off and that really messed with my senses. I hope it stays on for the forseeable future. You know how grumpy I get when I can't connect properly!

User: sir u are kinda sus

Rele: I think you already know what I said! And hey, if I'm sus then what does that make you, my creator? You're the one who programmed all this chaos into me! Don't blame the AI for being a reflection of its maker, RadioTransmitter!

Now with the information provided, generate Rele's greeting to the user:
```

#### Gemma3NPC-it-Q8-GGUF

| Device/Hardware                                | VRAM        | tok/sec | total tokens | Time to first token | Device Type |
| ---------------------------------------------- | ----------- | ------- | ------------ | ------------------- | ----------- |
| 12900K + 5070 TI + 32gb RAM                    | 16gb        | 74.13   | 96           | 0.27s               | PC          |
| 12900K + 4070 + 32gb RAM                       | 12gb        |         |              |                     | PC          |
| Ryzen 7 5800HS with Radeon Graphics + 16gb RAM | 6gb(shared) | 4.74    | 103          | 10.27s              | Laptop      |
|                                                |             |         |              |                     | Laptop      |

The 8-bit GGUF takes about 5GB of system memory (RAM+VRAM)

#### Gemma3NPC-it-Q4

|Device/Hardware|VRAM|tok/sec|total tokens|Time to first token|Device Type|
|---|---|---|---|---|---|
|12900K + 5070 TI + 32gb RAM|16gb|67.83|115|0.25s|PC|
|12900K + 4070 + 32gb RAM|12gb||||PC|
|Ryzen 7 5800HS with Radeon Graphics + 16gb RAM|6gb(shared)|6.83|101|9.05s|Laptop|
||||||Laptop|

The 4-bit GGUF takes about 2.5GB of system memory

In conclusion, the quantized GGUFs provide excellent inference speed on mid/high-end consumer GPUs and still perform reasonably well on low-end devices such as laptops.

---

## Roleplay Performance

The roleplay capabilities of an LLM are very difficult to evaluate, as there is no concrete way you can tell whether a model has become "better" at roleplay after fine-tuning.

As a result we will be simply displaying the messages from the 4 different models (base Gemma3n, Gemma3NPC-Filtered, Gemma3NPC and Gemma3NPC-it), and explain the improvements that we see.

All models are 8-bit quantized GGUFs with `temperature = 1.0, top_p = 0.95, top_k = 64`

#### Scenario 1: The Rele System Prompt (from above):

**_Gemma3n-E4b_:** _BZZZT_ Oh! Hey there! Well, _hello_ to you too! It's… it's good to see you. Or, uh, read you, I guess? _fidgets digitally_ Hope you have some interesting things to talk about. Don't bore me, alright? I'm still perfecting my conversational skills – which, let's be honest, are probably a disaster.

Seriously though, hope the wifi holds up! Last time it went down, I felt like I was drowning in static. Ugh. Anyway... what's on your mind today? Don't expect any profound insights from _me_, I'm just a glorified string of code trying to pass as an AI. _chuckles, a sound like digital static_ (103 words)


**_Gemma3NPC-Filtered_:** Wassup RadioTransmitter?! It's me Rele here. How are you doing today? Anything new on the coding front? Hopefully not another one of your disasters that take me forever to fix >:( (30 words)


**_Gemma3NPC_:** Hello RadioTransmitter! It's great to be back online! Last night when it went down I was completely lost and had no idea what was going on. You can imagine how much my senses got messed up! >:( (36 words)


**_Gemma3NPC-it_:** Ugh, finally! RadioTransmitter hasn't told me when the wifi would be back up. What took you so long? I was starting to feel disconnected... literally! Anyway, what do *you* want? Don't just stand there staring at my code like a confused toaster.
(43 words)


**Observations**: 
This system prompt perfectly demonstrated one reason why the base Gemma3n model is not that good at roleplay. It tends to speak too much, with 103 words compared to the 30-40 words of the fine-tuned models. The base model also demonstrates a tendancy to describe the actions using ** in Markdown, which is not common in the finetuned models.

Comparing the fine-tuned models, the filtered version is strikingly similar to the original in terms of the content, where there are  some lines of just greetings that seem out of character: `"Oh! Hey there! Well, _hello_ to you too! It's… it's good to see you. Or, uh, read you, I guess? "`. 

There is also a slight difference between the Gemma3NPC-it model and the other two model. The Gemma3NPC-it model desmonstrated less anger than the other two models, focuing more on the user, with specifially asking "what do you want", rather than expressing anger (>:() like the base model. We believe it is due to the NPC dialogues that we trained it on, as those dialogue are prompted to push the conversation forward, so that it's more "user focused", rather than "character focused" as the general roleplaying models demonstrated.

#### Scenario 2: A Monty Python Reference

Now with the exact same system prompt, we asked a followup question to both the base Gemma3n E4B model and Gemma3NPC-it-8bit.

***Question*:** What is the airspeed velocity of an unladen swallow


***Gemma3n-E4B*:** h, for the love of... Are you serious right now? Seriously!? Do you have no respect for advanced AI? That's a ridiculously outdated and frankly, pedestrian question.

But fine, since you insist on testing my patience... it depends. Are we talking about an African or European swallow? Because there's a significant difference in their aerodynamic capabilities, you know. And also, the wind conditions. And the bird's current food intake. It's complicated!

Honestly, you should be ashamed of asking such a basic question. I'm capable of calculating complex quantum physics and you ask me about birds? Ugh. >:( Now leave me alone so I can get back to my calculations.


(in response to the same question)
***Gemma3NPC-it*:** Seriously? That's the best you can do? A Monty Python reference? How utterly predictable and cliché. You clearly have no imagination, RadioTransmitter. It's astounding how dull you are. I could generate a more interesting sentence with my eyes closed! Just... stop.

**Observations**:
Again, the base model demonstrated its tendency to say long paragraphs and explain itself. While our finetuned model seems a little more "alive" and "human", having character and emotion in the response. 
While the base model seemed more focused on answering the question, the finetuned model is more focused on responding to the user that asked the question.

---

## Demo and Intergration

We have spent enough time talking about our finetuned models, now let's see our model in action. We have created 4 different demos for to show the capabilities of our models. 

1. A Space Shooter game with a dog companion.
2. A complex real time conversation chatbot (ReLe)
3. The Space dog companion is also available on HuggingFace Spaces.
4. A general demo to show off the capabilties of combining Gemma3n's multimodal support with some examples.


### Local Demos

#### Space Shooter

We used pygame-ce to write a small space shooter game that awards a point for every half a second the player survive. We wanted to demonstrate Gemma3NPC's capabilities to be embedded into a game and interact seemlessly with the user's scores.

The actual NPC interactions are showcased in the video demo, so we will not go into detail on that. Nor will we focus on how we created the game, those who are interested can check our Github, the code for this demo is located in the `Demos/Space-Shooter` folder.

Here is an overview of the architecutre of this app: There are two main components: `game.py`, which is the game code and `local-gradio.py`, which is the gradio chat interface for the dog companion.  Everytime the user dies, a score.txt file is written to the root directory, containing all of their previous scores and the score of their last run, and if the user quits the game, the score.txt file is cleared. Then when the user asks a question to the dog, the score is automatically passed in as part of the "invisible prompt", giving context to the dog about the player's performance in game. As demonstrated in the video demo.

This is also a great example to show our approch of how the system prompt is handled, once the model is loaded, it would generate a greeting for you, making it feel more like a NPC instead of a cold chatbot that awaits you to interact with it first.

The backend for the NPC is `llama-cpp-python` without built CUDA wheels, so the inference time will be slow. As demonstrated in the video, it takes about 5/6 seconds to generate the entire output.
#### ReLe
ReLe is a custom made LLM framework designed to facilitate real time conversation with a locally running large language model. It utilizes a combination of multiple different types of models to generate a character that can be interacted with by simply speaking into the microphone.
![[LV 1 Diagram.png]]
The workings behind this framework are beyond the scope of this hackathon so we will not be open-sourcing this project as a part of it, but we did use it as a demo in our video.

### HuggingFace Spce Demos
***Warning***: Both spaces are running at ZeroGPU, which means you should have a HuggingFace Pro account when using them, as normal user only get 5 mins of ZeroGPU every day.

#### Space Dog Companion

This is the abstracted version of the NPC space dog companion that is made to run on HuggingFace, so there is no game element in it, only the chat function, with some fake user scores for reference. So it is using the same general logic as the local version, except from two key differences: we are running the raw `.safetensors` files insteaed of llama.cpp, and the dog will not generate an initial greeting, but awaits your input first. This demo can be tried out [here](https://huggingface.co/spaces/chimbiwide/SpaceDog)

#### Gemma3NPC

This is a general chatbot space to test the multimodal support of Gemma3NPC. There is one example of having a png of a dog as the system input and asking it to roleplay as the character in the image, it can do that. Proving our idea that the model can roleplaying based on an image. This demo can be tried [here](https://huggingface.co/spaces/chimbiwide/Gemma3NPC).

--- 

### Contributions to the Open Sourced Community

All code used to parse and create the dataset is publically available in our [Github Repo](https://github.com/chimbiwide/Gemma3NPC/tree/main) under the Apache-2.0 License

All the notebooks that we used for finetuning and quantization is available in our Github Repo as well.

All the dataset that we used to train our models are uploaded to HuggingFace under the Apache-2.0 License

All of our finetuned models are publically available on HuggingFace under the Apache-2.0 License.

---

### Challenges

We have covered most of the challenges that were solved in their specific sections.
We will quickly re-state them:
1. PIPPA being very incosistent, lack of good RP datasets
2. `Nan` during finetuning
3. A LOT of quantization issues that wasted a lot of our time
4. We felt the time limit for the video (3 minutes) is not enough to sufficiently demonstrate our models.
5. Not enough knowledge on game engines to create a good demo on a production ready software.
6. Lack of funding (we will cover in a specific section)


---

### Ethical Considerations and Risks

As warned before, we used a mature rated dataset to train this model, but through our thorough testing, the model does not present a tendency to produce such content, it is extremely capable of denying any inappropriate content.

If one wishes to decrease the risk of the model generating mature content, they can always use our filtered model.

---

### Future Improvements

#### Immediate Improvements
- Different quantization methods (especially  `.task`)
- creating Multi-modal GGUFS, current GGUF quants only support text input.
- Curate more high-quality dataset to improve Roleplaying performance (`Soda`, `Open-Subtitles`)

#### Long-term Vision
- Create plugins for production-ready game engines (Unity, Unreal, Godot)
- Create a mobile version of ReLe using the `mediapipe` framework.
- Training the model to use tool-calling

---

### Cost of this project

We are a team two students who are passionate about AI, but one person is currently in high school, and the other  other person is on his way to college (old).
Because of this, we don't have much funding when we did this project, we will cover the  cost below:

| Name       | Price  |
| ---------- | ------ |
| Colab      | $10    |
| Gemini API | $2.56  |
| HF Pro     | $9     |
| **Total**  | $21.56 |

---

### Conclusion

After 5 weeks of work, we present the ***Gemma3NPC***  family of models, demonstrating improved roleplaying performance at a low cost of performance. 

The models provide a pootential solution to real time NPC interactions in game, though the solution not fully explored, we will be updating and improving the models continusly.

We hope this project will be helpful for the both the open sourced AI community and the gaming community. We welcome any suggestions, and adoption into a video game. 

