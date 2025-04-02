# Indian Legal Text Summarization

This project focuses on summarizing Indian legal texts using the `facebook/bart-large-cnn` model from the Hugging Face `transformers` library. The dataset used is [Indian-Legal-Text-ABS](https://huggingface.co/datasets/Yashaswat/Indian-Legal-Text-ABS), and the model was trained for 75 epochs on 1600 legal documents.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model](#model)
- [Training](#training)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction
Legal documents in India are often lengthy and complex. This project aims to generate concise summaries of legal texts to improve accessibility and comprehension.

## Dataset
The dataset used for training is [Indian-Legal-Text-ABS](https://huggingface.co/datasets/Yashaswat/Indian-Legal-Text-ABS), which consists of Indian legal documents and their corresponding summaries.

## Model
We utilize the `facebook/bart-large-cnn` model for abstractive text summarization. This transformer-based model is pre-trained on large-scale summarization datasets and fine-tuned on Indian legal texts for improved domain-specific summarization.

## Training
- **Model:** `facebook/bart-large-cnn`
- **Epochs:** 75
- **Documents:** 1600
- **Optimizer:** AdamW
- **Loss Function:** Cross-entropy loss
- **Batch Size:** Suitable batch size depending on available GPU memory
- **Learning Rate:** Tuned for optimal performance

## Installation
To set up the environment, install the necessary dependencies:
```bash
pip install transformers torch datasets
```

## Usage
Load the trained model and tokenizer:
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

def summarize(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs.input_ids, max_length=200, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

text = "Your long legal text here."
summary = summarize(text)
print(summary)
```

## Results
The model generates concise and coherent summaries of Indian legal texts. Fine-tuning on domain-specific data improves the relevance and accuracy of the summaries.

## Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request.

## License
This project is licensed under the MIT License.

