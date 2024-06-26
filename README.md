### Fine-tuned Mistral 7B and prompting
* [Mistral 7B](https://mistral.ai/news/announcing-mistral-7b/) was selected as a base LLM because of it's performance when compared to Llama 2 family and it's model size. Although I didn't had much time to explore other LLM's, I think this was the only model which could be quantized to 4-bit and then loaded to a 3060-Ti (8GB vRAM).
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


#### Everything else that was used while working on this project, any resource that might be useful
0. https://www.youtube.com/results?search_query=train+a+named+entity+recognition+model
1. https://www.youtube.com/watch?v=uKPBkendlxw
2. https://www.youtube.com/watch?v=2XUhKpH0p4M
3. https://www.youtube.com/watch?v=ujubwa_oa-0
4. https://www.youtube.com/watch?v=sUtthdcPyhc
5. https://mistral.ai/news/announcing-mistral-7b/
6. https://duckduckgo.com/?t=ffab&q=spacy+named+entity+recognition+for+medical+text&ia=web
7. https://gbnegrini.com/post/biomedical-text-nlp-scispacy-named-entity-recognition-medical-records/
8. https://towardsdatascience.com/clinical-named-entity-recognition-using-spacy-5ae9c002e86f
9. https://gathnex.medium.com/mistral-7b-fine-tuning-a-step-by-step-guide-52122cdbeca8 (Use the medical dataset from here for finetuning, dataset: https://huggingface.co/datasets/gathnex/Gath_baize)
10. https://adithyask.medium.com/a-beginners-guide-to-fine-tuning-mistral-7b-instruct-model-0f39647b20fe (Dataset used: https://huggingface.co/datasets/TokenBender/code_instructions_122k_alpaca_style)
11. https://levelup.gitconnected.com/a-step-by-step-guide-to-runing-mistral-7b-ai-on-a-single-gpu-with-google-colab-274a20eb9e40
12. https://medium.com/@scholarly360/mistral-7b-complete-guide-on-colab-129fa5e9a04d
13.https://levelup.gitconnected.com/a-step-by-step-guide-to-runing-mistral-7b-ai-on-a-single-gpu-with-google-colab-274a20eb9e40
(Mistral on single gpu: https://webcache.googleusercontent.com/search?q=cache:https://levelup.gitconnected.com/a-step-by-step-guide-to-runing-mistral-7b-ai-on-a-single-gpu-with-google-colab-274a20eb9e40&sca_esv=578489342&strip=1&vwsrc=0)
