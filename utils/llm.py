# third-party imports
from langchain_community.llms.llamacpp import LlamaCpp
from huggingface_hub.file_download import hf_hub_download

# local imports
from .constants import MODELS


def create_model(model: str, temperature: float):

    repo_id = MODELS[model]['repo_id']
    filename = MODELS[model]['filename']

    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir='models/'
    )

    return LlamaCpp(
            model_path=f'models/{filename}',
            top_p=.95,
            n_ctx=2048,
            n_gpu_layers=-1,
            max_tokens=1024,
            temperature=temperature)
