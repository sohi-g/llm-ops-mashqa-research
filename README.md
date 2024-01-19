# Large Language Model Operations (LLMOps)

## MASHQA (Multiple Answer Spans Healthcare Question Answering) Research Case Study
🤗Hugging Face Model Link[sohi-g/MASHQA-Mistral-7B-Instruct]

## Overview
This project involves the fine-tuning of the Mistral-7B-Instruct model on the MASHQA dataset, with a focus on Healthcare Question Answering. The approach utilizes prompt engineering techniques and employs RAG-Retrieval Augmented Generation for enhanced performance.

## Objective
The primary goal of this project is to provide accurate and safe answers to user queries related to healthcare. By fine-tuning the Mistral-7B-Instruct model on the MASHQA dataset, the aim is to improve the model's ability to understand and respond to health-related questions effectively.

## Techniques Used
1. **Mistral-7B-Instruct Model:** The base model used for this project is Mistral-7B-Instruct, a powerful language model designed for instructive text comprehension.

2. **Prompt Engineering:** To optimize the model for healthcare-related questions, prompt engineering techniques are employed. This involves designing specific prompts that guide the model to focus on healthcare domains.

3. **RAG-Retrieval Augmented Generation:** The RAG framework is utilized to enhance question answering. By integrating a retrieval step before generation, the model leverages relevant information for more accurate and contextually appropriate responses.

## Dataset
The MASHQA dataset serves as the training and evaluation data for this project. This dataset is curated to include a diverse range of healthcare-related questions, ensuring the model's robustness in handling various health domains. The dataset can be found here[https://drive.google.com/file/d/1ism3N3kMapliaORZQaQU8obNycF8rH9p/view].

## Workflow
1. **Data Preparation:** The MASHQA dataset is preprocessed to extract relevant question-answer pairs along with context and formulate appropriate prompts for fine-tuning.

2. **Fine-tuning:** The Mistral-7B-Instruct model is fine-tuned on the prepared dataset, leveraging the QLoRA (Quantized Low-Rank Adaption) technique on GCP Vertex AI platform.

3. **RAG Integration:** The RAG framework is integrated into the model to enable retrieval-augmented generation during question answering.

4. **Evaluation:** The fine-tuned model is evaluated on the MASHQA dataset to measure its performance in providing accurate and safe healthcare-related answers.

## Results
The results of the fine-tuned Mistral-7B-Instruct model are assessed based on metrics such as answer accuracy, contextual relevance, and safety in healthcare information dissemination.

## Conclusion
This project aims to contribute to the field of Healthcare Question Answering by leveraging state-of-the-art language models and techniques. The fine-tuned Mistral-7B-Instruct model, enriched with prompt engineering and RAG, is designed to enhance the user experience in obtaining reliable and secure health-related information.

Feel free to use, modify, and contribute to this project. Feedback and contributions are welcome to further improve the model's capabilities and address new challenges in healthcare question-answering.
