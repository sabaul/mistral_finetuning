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




## Alternate Approaches
If I had more time, I would have experimented with some of the approaches below and maybe selected a combination of these.

#### Step 1: Named Entity Recognition
- **Train a custom BERT-Model with tagging relevant entities**
    - **Challenges:**
        - Getting the relevant data and tagging it is a challenging task.
        - Knowing which entities to tag requires domain knowledge and some professional expertise.
- **Use a pre-built solution like [Google's Healthcare NLP API](https://cloud.google.com/healthcare-api/docs/concepts/nlp)**, where the input is the text content and it can extract all the entities from the text. Filtering could be applied in the post processing of the output from the API to fit to our use case.
    - Due to the challenges in the custom BERT-Model, I would prefer to use a pre-built solution like the Healthcare NLP API as it should be able to extract almost all the relevant medical entities from the input text.
    - And some kind of post-processing logic could be used to filter out the required entities.
* **NOTE**: If the input is not a text file, and maybe a pdf. We could use some OCR techniques before passing the text entities to either the custom model or the NLP API. It will again depend on the kind of document format, but for the sake of simplicity, I am assuming here that the input is also a text file.


#### Step 2: Search the clinical document using a text query
- Ideally I would prefer to use a pre-built solution from HuggingFace models to generate a sentence embedding from the input text (after breaking up the contents into individual sentences).
- The custom model trained in step 1 could also be used for generating these embeddings.
- Store the generated sentence embeddings into NumPy array (Elastic Search or [Faiss](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/) could be used in case we want to build a large scale solution and search over the entire database of documents at hand).
- For the current assignment, I would have used a NumPy array only for storing the embeddings, as the user would be searching within the document shared by the individual only (an assumption that the search result needs to be only from the user's input document).
- Given that the user would be searching the clinical document, we could take an input from the user for the relevant entities that the individual wants to search for and use some kind of similarity metric (like cosine similarity) to get the sentence which is most similar to the query.
- In order to provide use some more context, with the most similar sentence to the user's input. We could also provide 1-2 sentences around that relevant sentence to provide some more context.


#### Step 3: Summarization
- I would have used Mistral-7B for the document summarization, it is small and should be capable of producing better summarization results if fine-tuned with more data and for some more time.


## How to Replicate and make inference
- To setup the environment, requirements.txt file is shared in the zip.
- The fine-tuned model parameters are shared in the Google drive, which could be used to load the model state which I have also used.
- In order to replicate the working notebook (inference.ipynb or make_inference.py) a sample input text file (sample_input.txt) is provided in the zip folder.
- To test on other files, we will need to either replace the content of *sample_input.txt* or create a new file and change the name in the code itself (notebook as well as .py file).
- **The user will need to give an input for the text query to search for similar sentences.**
- I would suggest to use the inference.ipynb notebook, as loading the model multiple times while using the .py file will take a lot of inference time. Whereas loading the model once and making inference multiple times using the notebook should be more efficient.