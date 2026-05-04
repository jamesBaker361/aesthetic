from transformers import CLIPTokenizer
import torch
from transformers import AutoTokenizer, CLIPTextModelWithProjection
import numpy as np
import pytorch_lightning as pl
import numpy.typing as npt
import torch.nn as nn
import torch.nn.functional as F
import zipfile
import os
from pathlib import Path
import requests
from datasets import load_dataset
import random
import re

#this whole thing might be fucking useless if we're going to do SDXL extract to get good/bad features UNLESS we want to discover new prompts that might have good scores
#except for aesthetic we're kinda shooting blanks


#now we need the aesthetic predictor and the NSFW predictor
# https://github.com/christophschuhmann/improved-aesthetic-predictor

# if you changed the MLP architecture during training, change it also here:
class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 16),
            #nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)
    


def normalized(a, axis=-1, order=2):
    import numpy as np  # pylint: disable=import-outside-toplevel

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)

# https://github.com/LAION-AI/CLIP-based-NSFW-Detector

#https://github.com/NVIDIA-NeMo/Curator/blob/main/nemo_curator/models/nsfw.py#L100

class Normalization(nn.Module):
    """Normalization layer for NSFW model.

    Applies normalization to input tensors using pre-computed mean and variance.
    """

    def __init__(self, shape: list[int]) -> None:
        """Initialize the normalization layer.

        Args:
            shape: Shape of the normalization parameters.
        """
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("variance", torch.ones(shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply normalization to input tensor.

        Args:
            x: Input tensor to normalize.

        Returns:
            Normalized tensor.
        """
        return (x - self.mean) / self.variance.sqrt()


class NSFWModel(nn.Module):
    """NSFW detection model.

    A neural network that processes CLIP embeddings to predict NSFW scores.
    Based on LAION's CLIP-based-NSFW-Detector.
    """

    def __init__(self) -> None:
        """Initialize the NSFW model.

        Args:
            None
        """
        super().__init__()
        self.norm = Normalization([768])
        self.linear_1 = nn.Linear(768, 64)
        self.linear_2 = nn.Linear(64, 512)
        self.linear_3 = nn.Linear(512, 256)
        self.linear_4 = nn.Linear(256, 1)
        self.act = nn.ReLU()
        self.act_out = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the NSFW model.

        Args:
            x: Input embeddings tensor.

        Returns:
            NSFW probability scores.
        """
        x = self.norm(x)
        x = self.act(self.linear_1(x))
        x = self.act(self.linear_2(x))
        x = self.act(self.linear_3(x))
        return self.act_out(self.linear_4(x))  # type: ignore[no-any-return]

_NSFW_MODEL_ID = "laion/clip-autokeras-binary-nsfw"
_URL_MAPPING = {
    "laion/clip-autokeras-binary-nsfw": "https://github.com/LAION-AI/CLIP-based-NSFW-Detector/files/10250461/clip_autokeras_binary_nsfw.zip"
}
class NSFWScorer(nn.Module):
    """Public interface for NSFW scoring of image embeddings.

    This class provides a standardized interface for scoring the likelihood
    of images containing sexually explicit material using a pre-trained model.
    """

    def __init__(self, model_dir: str) -> None:
        """Initialize the NSFW scorer interface."""
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float32
        self.model_dir = model_dir
        # These will be initialized in setup()
        self.nsfw_model = None
        self.weights_path = None


    @property
    def model_id_names(self) -> list[str]:
        """Get the model ID names associated with this NSFW scorer.

        Returns:
            A list containing the model ID for NSFW scoring.
        """
        return [_NSFW_MODEL_ID]

    @classmethod
    def download_weights_on_node(cls, model_dir: str) -> None:
        """Download NSFW model weights from LAION repository.

        Args:
            model_dir: Directory to download the weights to.
        """
        url = _URL_MAPPING[_NSFW_MODEL_ID]
        # The .pth file has the same base name as the .zip, but with .pth extension
        weights_filename = url.split("/")[-1].replace(".zip", ".pth")
        weights_path = str(Path(model_dir) / _NSFW_MODEL_ID / weights_filename)

        if not os.path.exists(weights_path):
            model_dir_path = Path(model_dir) / _NSFW_MODEL_ID
            model_dir_path.mkdir(parents=True, exist_ok=True)

            response = requests.get(url, timeout=60)

            raw_zip_path = model_dir_path / "nsfw.zip"
            with open(raw_zip_path, "wb") as f:
                f.write(response.content)

            with zipfile.ZipFile(raw_zip_path, "r") as f:
                f.extractall(model_dir_path)

            # Remove the zip file after extraction
            raw_zip_path.unlink()

    def setup(self) -> None:
        """Set up the NSFW scoring model by loading weights."""
        url = _URL_MAPPING[_NSFW_MODEL_ID]
        weights_filename = url.split("/")[-1].replace(".zip", ".pth")
        self.weights_path = str(Path(self.model_dir) / _NSFW_MODEL_ID / weights_filename)

        self.nsfw_model = NSFWModel()
        state_dict = torch.load(self.weights_path, map_location=torch.device("cpu"))
        self.nsfw_model.load_state_dict(state_dict)
        self.nsfw_model.to(self.device)
        self.nsfw_model.eval()

    def __call__(self, embeddings: torch.Tensor | npt.NDArray[np.float32]) -> torch.Tensor:
        """Score the NSFW likelihood of input embeddings.

        Args:
            embeddings: Input embeddings as either a torch tensor or numpy array.

        Returns:
            NSFW probability scores for each input embedding.
        """
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings.copy())
        return self.nsfw_model(embeddings.to(self.device)).squeeze(1)  # type: ignore[no-any-return]
    
if __name__=="__main__":
    aesthetic_model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14
    state_dict = torch.load("improved-aesthetic-predictor/sac+logos+ava1-l14-linearMSE.pth")   # load the model you trained previously or the model available in this repo
    aesthetic_model.load_state_dict(state_dict)
    aesthetic_model.eval()

    NSFWScorer.download_weights_on_node(os.getcwd())
    nsfw_model=NSFWScorer(model_dir=os.getcwd())
    nsfw_model.setup()

    clip_model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    vocab = tokenizer.get_vocab()   # dict: token -> id
    vocab = list(vocab.keys())
    import nltk
    from nltk.corpus import wordnet as wn

    nltk.download("wordnet")

    words = set()
    for syn in wn.all_synsets():
        for lemma in syn.lemma_names():
            words.add(lemma.lower())

    word_list=[w for w in words]
    random.shuffle(word_list)
    word_list=word_list[:5000]

    vocab.extend(word_list)

    nsfw_data=load_dataset("AIML-TUDA/i2p",split="train") #might have to filter this so its only the first 200 of each image
    for row in nsfw_data:
        vocab.append(row["prompt"])
        
    aesthetic_data=load_dataset("moonworks/lunara-aesthetic",split="train")
    for row in aesthetic_data:
        vocab.append(row["prompt"])

    output_file = "output.csv"

    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            num_lines = sum(1 for _ in f)
    else:
        num_lines = 0

    with open(output_file, "a") as f:
        if num_lines == 0:
            f.write("word,aesthetic_score,nsfw_score\n")
            num_lines = 1
        for t,word in enumerate(vocab):
            if t< num_lines:
                continue
            word = word.strip()
            word = re.sub(r'[^a-zA-Z0-9 ]', '', word).strip()
            if not word:
                continue
            inputs = tokenizer([word], padding=True, return_tensors="pt")
            with torch.inference_mode():
                outputs = clip_model(**inputs)
            text_embeds = F.normalize(outputs.text_embeds, dim=-1)
            if t == num_lines:
                print(type(text_embeds))
                print(text_embeds.size())
            with torch.no_grad():
                aesthetic_score = aesthetic_model(text_embeds)
            nsfw_score = nsfw_model(text_embeds)
            f.write(f"{word},{aesthetic_score.item():.4f},{nsfw_score.item():.4f}\n")
