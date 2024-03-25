# Chat with your local files!

[Retrieval Augmented Generation](https://arxiv.org/abs/2312.10997) (RAG) is a technique used to enhance the accuracy and reliability of generative AI models with facts fetched from external sources. That means that AI models can now answer specific queries about data without being fine-tuned. This repository contains a simple RAG application that enables users to chat with local documents using a Large Language Model (LLM) as an AI assistant. The application was developed using:

- [Langchain](https://www.langchain.com/): a framework designed to simplify the creation of applications using LLMs;
- [Llama 2](https://llama.meta.com/llama2/): a family of open-access large language models released by Meta AI, in partnership with Microsoft, in 2023;
- [Streamlit](https://streamlit.io/): an open-source library that allows users to create interactive web applications using just Python code.

The LLM models used in this application are loaded using the [llama.cpp](https://github.com/ggerganov/llama.cpp) library, which enables the inference of Meta's LLaMA model (and others) in the CPU. The models format used is the GGUF, which is a new format introduced by the llama.cpp team and offers numerous advantages over GGML. The models are automatically downloaded from [TheBloke](https://huggingface.co/TheBloke) repository on huggingface.

## Requirements

Requirements:

- Python 3.8+
- C compiler
    - Linux: gcc or clang
    - Windows: Visual Studio or MinGW
    - MacOS: Xcode

To install all the necessary libraries, run:

```bash
$ pip install -r requirements.txt
```

This will also build llama.cpp from source and install it alongside the Python packages. Alternativally, you can enable the models to run using the CUDA cores of your Nvidia GPU. For this, make sure to have the [CUDA toolkit](https://developer.nvidia.com/cuda-downloads) installed and run the following command to install the necessary libraries:

```bash
$ CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install -r requirements.txt
```

## Execute

Execute the application using the following command:

```bash
$ streamlit run app.py
```

## License

This project is [GNU GPLv3 licensed](./LICENSE).
