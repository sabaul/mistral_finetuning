## Current Approach

#### Fine-tuned Mistral 7B and prompting
* I chose [Mistral 7B](https://mistral.ai/news/announcing-mistral-7b/) as a base LLM because of it's performance when compared to Llama 2 family and it's model size. Although I didn't had much time to explore other LLM's, I think this was the only model which could be quantized to 4-bit and then loaded to a 3060-Ti (8GB vRAM).
* Mistral 7B is a 7.3B parameter model that:
    * Outperforms Llama 2 13B on all benchmarks
    * Outperforms Llama 1 34B on many benchmarks
    * Approaches CodeLlama 7B performance on code, while remaining good at English tasks
* The model was fine-tuned on [Gath_baize](https://huggingface.co/datasets/gathnex/Gath_baize) dataset. It is a conversation dataset constructed between Human and AI assisatance named Gathnex. It consists of 210k such conversations, it also contains medical conversations.
* So in order to fine-tune the Mistral 7B on medical dataset, I chose to filter out the medical conversations from the entire dataset and train the Mistral 7B on it. There are around 47k medical conversations extracted from the entire dataset.
* The **named entity extraction,** **searching the clinical document using a text query** and the **clinical document summarization**, prompts are used in order to generate the outputs.
* The Mistral 7B was fine-tuned for 20 epochs.
* The model behaviour on named entity extraction was not consistent, so sometimes it predicts entities, sometimes some small phrases which contains the entities.
* The text query search was not performing well when a single output was required, so the prompt is designed such that we extract a set of similar sentences, this gives us a set of sentences which might be similar to the user's query.
* The model performance on summarization is good.
