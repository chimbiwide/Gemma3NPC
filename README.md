# Gemma3NPC--A solution for live NPC interactions

New model -- GemmaReLe

We experimented with a less conservative training parameters!

Check out `Gemma3NPC_Instruct_Beta.ipynb`, inside the `Training` folder.   
The actual model is available on [HuggingFace](https://huggingface.co/collections/chimbiwide/gemma3npc-it-beta)

> [!NOTE]
> Currently all notebooks in the `Training` folder cannot be rendered. Showing `Invalid Notebook`     
> It will work if the notebook is opened in an IDE or Google Colab.      
> Scroll down to see the changelog

We added a folder containing the code used to generate our `NPC-Dialogue_v2` dataset. Visit its [HF repo](https://huggingface.co/datasets/chimbiwide/NPC-Dialogue_v2) for more info.

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

---

### Repository structure

This repository is split into three folders: `Datasets`, `Demos` and `Training`. Containing python code for each part of the hackathon. 

`Datasets` contains the code to create the four different datasets that we created to train the model.

`Demos` contains the code that was used to create the two Huggingface spaces.

`Training` contains the Notebooks that we used to finetune Gemma3N

---

### Changelog

##### 8/13/2025

Realized that we uploaded a text file for the Gemma3NPC-Filtered notebook. The original notebook is still there and can work by modifying the file extension.    
We uploaded an actual notebook as well.

##### 10/13/2025

Updated the code to now include NPC-Dialogue_v2, our new and improved version of NPC_dialogue, with all the scripts included. We are actively working on Gemma3NPC_v2. 

##### 11/25/2025

Added the training notebook for `Gemma3NPC-it-beta`.

##### 11/26/2025

Added folder `ReLe_Syn` for the python scripts used to correctly format [KeeganC/ReLe_Synthetic_v1](https://huggingface.co/datasets/KeeganC/ReLe_Synthetic_v1)

Added a new training notebook [GemmaReLe](https://github.com/chimbiwide/Gemma3NPC/blob/main/Training/GemmaReLe.ipynb) for the GemmaReLe model.
