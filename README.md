# GemmaNPC — A solution for live NPC interactions

**Latest model**: [Gemma4NPC-E4B](https://huggingface.co/chimbiwide/Gemma4NPC-E4B) — fine-tuned on Gemma 4-E4B with the new [RolePlay-NPC-Quest](https://huggingface.co/datasets/chimbiwide/RolePlay-NPC-Quest) dataset.

Previous generation: [Gemma3NPC-1b](https://huggingface.co/collections/chimbiwide/gemma3npc-1b) | [Gemma3NPC-it](https://huggingface.co/collections/chimbiwide/gemma3npc-it-688ea87f8773403b845996c1)

## Introduction

Video games have grown bigger and bigger over the years: _Skyrim_, _The Witcher 3_, _Dark Souls_, _Cyberpunk 2077_, _Baldur's Gate 3_. They have all taken the same approach of scripting the various dialogues in games, with _Baldur's Gate 3_ featuring about 1.365 million words of dialogue according to [IGN](https://www.ign.com/articles/baldurs-gate-3s-collective-dialogue-has-triple-the-word-count-of-the-lord-of-the-rings-larian-reveals). While these hand-written scripts may seem relatively immersive the first time around, when players replay the game multiple times, or simply play it for long enough, the repetitiveness starts to become apparent. Many complaints of this nature can be found easily online, which shows this is an ongoing issue in video games.

With the rise of AI, there are already talks of powering NPCs with large language models, which would be able to create a more immersive world for the player, allowing them to have extended interactions with the NPCs, making them more "lifelike", instead of scripted machines repeating the same lines. This can revolutionize the video game industry, with newer ways of approaching NPC creation in games.

Existing solutions can be found, such as the `NPC-LLM`s created by [Gigax](https://huggingface.co/Gigax). However, these models have issues: they are either too big (the models are 7/8Bs), requiring extensive hardware, or don't really show the details on how the models are created, making developers struggle when trying to implement them in their games.

This is why GemmaNPC was created. The latest generation, **Gemma4NPC-E4B**, is fine-tuned on Google's new **Gemma 4-E4B** model. Like its predecessor Gemma3NPC, the quantized versions are small enough to run on low-end machines at a reasonable speed, while also retaining multimodal support for text, image, and audio input. Imagine an LLM-powered NPC that can respond to your text, recognize your outfit through an in-game image input, or even let you speak with it — the possibilities are only limited by imagination and compute.

GemmaNPC is not intended to replace traditional scripted NPCs, but merely as an extension for those interested in creating a more immersive and "lifelike" NPC. It benefits everyone: game developers can add it to their games to create better NPCs or use it to generate dialogue, while players can enjoy a way more immersive gaming experience at a low cost of performance.

All GemmaNPC models provide a raw fp16 model that can be run using `transformers` or `vLLM`, and a 4-bit and an 8-bit quantized GGUF for directly embedding into a game using Llama.cpp or any other compatible inference engine.

---

### Datasets

Gemma4NPC-E4B is trained on **[RolePlay-NPC-Quest](https://huggingface.co/datasets/chimbiwide/RolePlay-NPC-Quest)**, our most comprehensive dataset to date — a combination of:

- [PIPPA](https://huggingface.co/datasets/chimbiwide/pippa) — a large, high-quality RP conversation dataset
- [NPC-Quest-Dialogue](https://huggingface.co/datasets/chimbiwide/NPC-Quest-Dialogue) — our new quest-focused NPC dialogue dataset (first used in this generation)
- [NPC-Dialogue_v2](https://huggingface.co/datasets/chimbiwide/NPC-Dialogue_v2) — our second-generation synthetic NPC dialogue dataset

Previous Gemma3NPC generations used other variations of these datasets, the exact datasets are listed in their model cards.

---

### Model Variants

| Model | Base | Description |
|---|---|---|
| `Gemma4NPC-E4B` | Gemma 4-E4B | **Latest** — NPC fine-tune on RolePlay-NPC-Quest |
| `Gemma3NPC-it` | Gemma 3n-E4B | NPC-specific instruct fine-tune |
| `Gemma3NPC` | Gemma 3n-E4B | Base RP fine-tune |
| `Gemma3NPC-Filtered` | Gemma 3n-E4B | Filtered dataset variant (deprecated) |
| `Gemma3NPC-1b` | Gemma 3n-E1B | 1B parameter variant |

---

### Ethical Considerations and Risks

As warned before, we used a mature-rated dataset to train this model, but through our thorough testing, the model does not present a tendency to produce such content and is extremely capable of denying any inappropriate content.

If one wishes to decrease the risk of the model generating mature content, they can always use our filtered model.

---

### Future Improvements

#### Immediate Improvements
- Different quantization methods (especially `.task`)
- Multi-modal GGUFs — current GGUF quants only support text input
- Curate more high-quality datasets to improve roleplaying performance

#### Long-term Vision
- Create plugins for production-ready game engines (Unity, Unreal, Godot)
- Create a mobile version using the `mediapipe` framework
- Training the model to use tool-calling

---

### Cost of this project

We are a team of two students who are passionate about AI — one currently in high school, and the other a freshman in college.
Because of this, we don't have much funding. We cover the cost below:

| Name       | Price  |
| ---------- | ------ |
| Colab      | $10    |
| Gemini API | $2.56  |
| DeepSeek API | $7.89 |
| Colab | $30 |
| **Total**  | $21.56 |

---

### Conclusion

We are dedicated to the ongoing development, we present the **GemmaNPC** family of models, with the latest **Gemma4NPC-E4B** demonstrating improved roleplaying performance at a low cost of performance.

The models provide a potential solution to real-time NPC interactions in games. We will be updating and improving the models continuously.

We hope this project will be helpful for both the open-source AI community and the gaming community. We welcome any suggestions and adoption into a video game.

---

### Repository Structure

This repository is split into three folders: `datasets`, `demos`, and `training`.

`datasets` contains the code to create the datasets used to train the models.

`demos` contains the code used to create the two Hugging Face Spaces.

`training` contains the notebooks used to fine-tune each model generation.

---

### Changelog

##### 4/8/2026

Released **Gemma4NPC-E4B**, fine-tuned on Google's Gemma 4-E4B model using the new [RolePlay-NPC-Quest](https://huggingface.co/datasets/chimbiwide/RolePlay-NPC-Quest) dataset. This is the first GemmaNPC model to use the NPC-Quest-Dialogue dataset. Added training notebook under `training/Gemma4NPC/`.

##### 11/26/2025

Added folder `rele_syn` for the python scripts used to correctly format [KeeganC/ReLe_Synthetic_v1](https://huggingface.co/datasets/KeeganC/ReLe_Synthetic_v1).

Added a new training notebook [GemmaReLe](https://github.com/chimbiwide/Gemma3NPC/blob/main/training/GemmaReLe.ipynb) for the GemmaReLe model.

##### 11/25/2025

Added the training notebook for `Gemma3NPC-it-beta`.

##### 10/13/2025

Updated the code to now include NPC-Dialogue_v2, our new and improved version of NPC_dialogue, with all the scripts included.

##### 8/13/2025

Realized that we uploaded a text file for the Gemma3NPC-Filtered notebook. The original notebook is still there and can work by modifying the file extension. We uploaded an actual notebook as well.
