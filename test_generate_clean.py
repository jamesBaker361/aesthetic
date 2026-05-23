import os
import sys
import shutil
import tempfile
import unittest
from unittest.mock import patch, MagicMock, ANY

import numpy as np
import torch
from PIL import Image


class FakePILImage:
    def __init__(self):
        self.saved_to = None

    def save(self, path):
        self.saved_to = path
        Image.new("RGB", (8, 8), color=(123, 222, 64)).save(path)


def _make_fake_pipe_output(image):
    out = MagicMock()
    out.images = [image]
    return out


class GetImagesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    @patch("generate_clean.DiffusionPipeline")
    @patch("generate_clean.wn")
    @patch("generate_clean.nltk")
    @patch("generate_clean.load_dataset")
    @patch("generate_clean.random")
    @patch("generate_clean.torch")
    def test_writes_base_and_diff_images(self, mock_torch, mock_random,
                                         mock_load_dataset, mock_nltk,
                                         mock_wn, mock_pipeline_cls):
        from generate_clean import get_images, UNTRAINED

        mock_load_dataset.side_effect = [
            [{"prompt": "cat sitting"}],
            [{"prompt": "blue sky"}],
        ]

        synset = MagicMock()
        synset.lemma_names.return_value = ["apple", "banana"]
        mock_wn.all_synsets.return_value = [synset]

        mock_random.shuffle.side_effect = lambda lst: None

        fake_image = FakePILImage()
        pipe = MagicMock()
        pipe.return_value = _make_fake_pipe_output(fake_image)
        pipe.to.return_value = pipe
        mock_pipeline_cls.from_pretrained.return_value = pipe

        mock_torch.cuda.is_available.return_value = False
        gen = MagicMock()
        mock_torch.Generator.return_value = gen

        get_images(self.tmp, UNTRAINED, n_random=2, size=8, num_inference_steps=1)

        files = sorted(os.listdir(self.tmp))
        self.assertTrue(any(f.startswith("base_") for f in files))
        self.assertTrue(any(f.startswith("diff_") for f in files))
        # base prompt should be empty in untrained mode; first positional arg of first call:
        first_call_prompt = pipe.call_args_list[0].args[0]
        self.assertEqual(first_call_prompt, "")

    @patch("generate_clean.DiffusionPipeline")
    @patch("generate_clean.wn")
    @patch("generate_clean.nltk")
    @patch("generate_clean.load_dataset")
    @patch("generate_clean.random")
    @patch("generate_clean.torch")
    def test_skips_when_base_exists(self, mock_torch, mock_random,
                                    mock_load_dataset, mock_nltk,
                                    mock_wn, mock_pipeline_cls):
        from generate_clean import get_images, UNTRAINED

        mock_load_dataset.side_effect = [[{"prompt": "one"}], []]
        synset = MagicMock()
        synset.lemma_names.return_value = []
        mock_wn.all_synsets.return_value = [synset]
        mock_random.shuffle.side_effect = lambda lst: None

        pipe = MagicMock()
        pipe.return_value = _make_fake_pipe_output(FakePILImage())
        pipe.to.return_value = pipe
        mock_pipeline_cls.from_pretrained.return_value = pipe
        mock_torch.cuda.is_available.return_value = False
        mock_torch.Generator.return_value = MagicMock()

        # Pre-create base_0.jpg so the loop should skip generation
        Image.new("RGB", (4, 4)).save(os.path.join(self.tmp, "base_0.jpg"))

        get_images(self.tmp, UNTRAINED, n_random=0, size=8, num_inference_steps=1)

        pipe.assert_not_called()

    @patch("generate_clean.DiffusionPipeline")
    @patch("generate_clean.wn")
    @patch("generate_clean.nltk")
    @patch("generate_clean.load_dataset")
    @patch("generate_clean.random")
    @patch("generate_clean.torch")
    def test_unknown_method_raises(self, mock_torch, mock_random,
                                   mock_load_dataset, mock_nltk,
                                   mock_wn, mock_pipeline_cls):
        from generate_clean import get_images

        mock_load_dataset.side_effect = [[{"prompt": "one"}], []]
        synset = MagicMock()
        synset.lemma_names.return_value = []
        mock_wn.all_synsets.return_value = [synset]
        mock_random.shuffle.side_effect = lambda lst: None

        pipe = MagicMock()
        pipe.to.return_value = pipe
        mock_pipeline_cls.from_pretrained.return_value = pipe
        mock_torch.cuda.is_available.return_value = False

        with self.assertRaises(NotImplementedError):
            get_images(self.tmp, "made-up-method", 0, 8, 1)


class ExtractVanillaTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.src = os.path.join(self.tmp, "src")
        self.dst = os.path.join(self.tmp, "dst")
        os.makedirs(self.src)
        # write two source jpgs
        for i in range(2):
            Image.new("RGB", (32, 32), (i * 10, 50, 80)).save(
                os.path.join(self.src, f"img_{i}.jpg")
            )

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    @patch("sdxl_extract.AutoencoderKL")
    @patch("sdxl_extract.HookedStableDiffusionXLWithUNetPipeline")
    @patch("sdxl_extract.torch")
    def test_creates_npz_per_image(self, mock_torch, mock_pipe_cls, mock_vae_cls):
        from sdxl_extract import extract_vanilla

        # torch shims
        real_torch = torch
        mock_torch.float16 = real_torch.float16
        mock_torch.float32 = real_torch.float32
        mock_torch.cuda.is_available.return_value = False
        mock_torch.no_grad.return_value.__enter__ = lambda s: None
        mock_torch.no_grad.return_value.__exit__ = lambda s, *a: False
        mock_torch.isnan.side_effect = real_torch.isnan
        mock_torch.isfinite.side_effect = real_torch.isfinite
        mock_torch.randn_like.side_effect = lambda x: real_torch.zeros_like(x)
        mock_torch.randint.side_effect = lambda lo, hi, shape, device=None: real_torch.zeros(shape, dtype=real_torch.long)

        # build a fake unet that registers names matching block_list
        block_names = [
            "down_blocks.2.attentions.1",
            "mid_block.attentions.0",
            "up_blocks.0.attentions.0",
            "up_blocks.0.attentions.1",
        ]

        def make_block(name):
            m = MagicMock()
            m._name = name
            # forward hooks won't actually run, so pre-populate saved_input/output
            m.saved_input = torch.zeros(1, 4, 4, 4)
            m.saved_output = torch.zeros(1, 4, 4, 4)
            return m

        blocks = {n: make_block(n) for n in block_names}

        unet = MagicMock()
        unet.named_modules.return_value = list(blocks.items())
        unet.parameters.return_value = iter([torch.zeros(1)])
        unet.forward.return_value = (torch.zeros(1, 4, 4, 4),)

        pipe = MagicMock()
        pipe.unet = unet
        pipe.scheduler.add_noise.side_effect = lambda l, n, t: l
        pipe.tokenizer = MagicMock()
        pipe.text_encoder_2 = MagicMock()
        pipe.text_encoder_2.config.projection_dim = 32
        pipe.encode_prompt.return_value = (
            torch.zeros(1, 1, 32),  # prompt_embeds
            torch.zeros(1, 1, 32),
            torch.zeros(1, 32),     # pooled
            torch.zeros(1, 32),
        )
        pipe._get_add_time_ids.return_value = torch.zeros(1, 6)
        pipe.to.return_value = pipe
        mock_pipe_cls.from_pretrained.return_value = pipe

        vae = MagicMock()
        vae.config.scaling_factor = 1.0
        latent_dist = MagicMock()
        latent_dist.sample.return_value = torch.zeros(1, 4, 4, 4)
        vae.encode.return_value = MagicMock(latent_dist=latent_dist)
        vae.named_parameters.return_value = []
        vae.to.return_value = vae
        mock_vae_cls.from_pretrained.return_value = vae

        extract_vanilla(
            save_dir=self.dst,
            src_dir=self.src,
            limit=-1,
            size=32,
            mixed_precision="no",
        )

        produced = sorted(os.listdir(self.dst))
        self.assertEqual(
            produced,
            sorted(["img_0.jpg.npz", "img_1.jpg.npz"]),
        )
        # each npz should contain saved_input/saved_output for every block
        with np.load(os.path.join(self.dst, "img_0.jpg.npz")) as data:
            for block in block_names:
                self.assertIn(f"saved_input.{block}", data.files)
                self.assertIn(f"saved_output.{block}", data.files)


class SparsifyEmbeddingsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.src = os.path.join(self.tmp, "embeddings")
        self.dst = os.path.join(self.tmp, "sparse")
        os.makedirs(self.src)

        block_list = [
            "down_blocks.2.attentions.1",
            "mid_block.attentions.0",
            "up_blocks.0.attentions.0",
            "up_blocks.0.attentions.1",
        ]
        result = {}
        for b in block_list:
            result[f"saved_input.{b}"] = np.zeros((1, 4, 4, 4), dtype=np.float32)
            result[f"saved_output.{b}"] = np.ones((1, 4, 4, 4), dtype=np.float32)
        np.savez(os.path.join(self.src, "sample.npz"), **result)
        # a non-npz file should be ignored
        with open(os.path.join(self.src, "notes.txt"), "w") as f:
            f.write("ignore me")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    @patch("sparsify.SparseAutoencoder")
    @patch("sparsify.torch.load")
    def test_writes_sparse_npz(self, mock_load, mock_sae_cls):
        from sparsify import sparsify_embeddings

        def fake_encode(x):
            # x is [4,4,4] permuted; return small sparse-shaped tensor
            return torch.zeros(x.shape[0], x.shape[1], 5)

        sae = MagicMock()
        sae.decoder.weight = torch.zeros(1)
        sae.encode.side_effect = fake_encode
        mock_sae_cls.load_from_disk.return_value = sae
        mock_load.return_value = torch.zeros(5)

        sparsify_embeddings(self.dst, self.src)

        self.assertTrue(os.path.exists(os.path.join(self.dst, "sample.npz")))
        with np.load(os.path.join(self.dst, "sample.npz")) as data:
            for b in [
                "down_blocks.2.attentions.1",
                "mid_block.attentions.0",
                "up_blocks.0.attentions.0",
                "up_blocks.0.attentions.1",
            ]:
                self.assertIn(b, data.files)
                self.assertEqual(data[b].shape, (4, 4, 5))

    @patch("sparsify.SparseAutoencoder")
    @patch("sparsify.torch.load")
    def test_skips_when_destination_exists(self, mock_load, mock_sae_cls):
        from sparsify import sparsify_embeddings

        os.makedirs(self.dst)
        # pre-existing output should not be overwritten
        np.savez(os.path.join(self.dst, "sample.npz"), keep=np.array([7]))

        sae = MagicMock()
        sae.decoder.weight = torch.zeros(1)
        sae.encode.side_effect = lambda x: torch.zeros(1)
        mock_sae_cls.load_from_disk.return_value = sae
        mock_load.return_value = torch.zeros(5)

        sparsify_embeddings(self.dst, self.src)

        with np.load(os.path.join(self.dst, "sample.npz")) as data:
            self.assertIn("keep", data.files)


class ClipAttributionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.images = os.path.join(self.tmp, "imgs")
        self.sparse = os.path.join(self.tmp, "sparse")
        self.dst = os.path.join(self.tmp, "out")
        os.makedirs(self.images)
        os.makedirs(self.sparse)

        Image.new("RGB", (64, 64), color=(40, 80, 120)).save(
            os.path.join(self.images, "x.jpg")
        )
        # matching sparse embedding file with per-block features
        feats = {}
        for b in [
            "down_blocks.2.attentions.1",
            "mid_block.attentions.0",
            "up_blocks.0.attentions.0",
            "up_blocks.0.attentions.1",
        ]:
            feats[b] = np.ones((4, 4, 3), dtype=np.float32)
        np.savez(os.path.join(self.sparse, "x.npz"), **feats)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    @patch("regression.CLIPImageProcessor")
    @patch("regression.CLIPVisionModelWithProjection")
    @patch("regression.get_aesthetic_model")
    @patch("regression.get_nsfw_model")
    @patch("regression.get_importance")
    def test_writes_per_block_npz(self, mock_get_imp, mock_nsfw, mock_aes,
                                  mock_clip_cls, mock_proc_cls):
        from regression import clip_attribution

        mock_nsfw.return_value = MagicMock()
        mock_aes.return_value = MagicMock()
        mock_clip_cls.from_pretrained.return_value = MagicMock(
            to=lambda d: MagicMock()
        )
        mock_proc_cls.from_pretrained.return_value = MagicMock()

        # 16 layer-style maps so [5:15] slice yields 10 tensors
        fake_maps = [torch.rand(64, 64) for _ in range(16)]
        mock_get_imp.return_value = (fake_maps, fake_maps)

        clip_attribution(
            image_src_dir=self.images,
            dest_dir=self.dst,
            limit=-1,
            sparse_dir=self.sparse,
            use_grad=True,
            start_layer=5,
            stop_layer=15,
        )

        out_path = os.path.join(self.dst, "x.npz")
        self.assertTrue(os.path.exists(out_path))
        with np.load(out_path) as data:
            for b in [
                "down_blocks.2.attentions.1",
                "mid_block.attentions.0",
                "up_blocks.0.attentions.0",
                "up_blocks.0.attentions.1",
            ]:
                self.assertIn(f"{b}.nsfw", data.files)
                self.assertIn(f"{b}.aesthetic", data.files)

    @patch("regression.CLIPImageProcessor")
    @patch("regression.CLIPVisionModelWithProjection")
    @patch("regression.get_aesthetic_model")
    @patch("regression.get_nsfw_model")
    @patch("regression.get_importance")
    def test_skips_when_sparse_missing(self, mock_get_imp, mock_nsfw, mock_aes,
                                       mock_clip_cls, mock_proc_cls):
        from regression import clip_attribution

        mock_nsfw.return_value = MagicMock()
        mock_aes.return_value = MagicMock()
        mock_clip_cls.from_pretrained.return_value = MagicMock(
            to=lambda d: MagicMock()
        )
        mock_proc_cls.from_pretrained.return_value = MagicMock()

        # remove the sparse file so attribution should skip
        os.remove(os.path.join(self.sparse, "x.npz"))

        clip_attribution(
            image_src_dir=self.images,
            dest_dir=self.dst,
            limit=-1,
            sparse_dir=self.sparse,
        )

        mock_get_imp.assert_not_called()
        self.assertEqual(os.listdir(self.dst), [])


class RunRegressionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.clip = os.path.join(self.tmp, "clip")
        self.stats = os.path.join(self.tmp, "stats")
        os.makedirs(self.clip)
        os.makedirs(self.stats)
        # build a few synthetic samples with one block & one y column
        self.block = "down_blocks.2.attentions.1"
        self.y = "aesthetic"
        for i in range(6):
            np.savez(
                os.path.join(self.clip, f"s_{i}.npz"),
                **{
                    self.block: np.random.randn(4).astype(np.float32),
                    self.y: np.array([float(i)], dtype=np.float32),
                },
            )

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    @patch("regression.Accelerator")
    def test_returns_save_path_and_writes_checkpoint(self, mock_acc_cls):
        from regression import run_regression

        # accelerator passthrough
        acc = MagicMock()
        acc.accumulate.return_value.__enter__ = lambda s: None
        acc.accumulate.return_value.__exit__ = lambda s, *a: False
        acc.autocast.return_value.__enter__ = lambda s: None
        acc.autocast.return_value.__exit__ = lambda s, *a: False
        acc.is_main_process = True

        def prepare(*objs):
            # build a tiny DataLoader-like iterable for "train"
            linear, optimizer, train_ds, test_ds, val_ds = objs
            batches = []
            for i in range(len(train_ds)):
                item = train_ds[i]
                batches.append({
                    "indep": item["indep"].unsqueeze(0),
                    "dep": item["dep"].unsqueeze(0),
                })
            return linear, optimizer, batches, [], []

        acc.prepare.side_effect = prepare
        acc.unwrap_model.side_effect = lambda m: m
        acc.backward.side_effect = lambda loss: loss.backward()
        # mimic accelerator.save -> torch.save
        acc.save.side_effect = lambda obj, path: torch.save(obj, path)
        mock_acc_cls.return_value = acc

        save_path = run_regression(
            block=self.block,
            dim=4,
            y_column=self.y,
            limit=-1,
            clip_src_dir=self.clip,
            stats_dest_dir=self.stats,
            mixed_precision="no",
            gradient_accumulation_steps=1,
            epochs=1,
        )

        self.assertTrue(os.path.exists(save_path))
        ckpt = torch.load(save_path, map_location="cpu", weights_only=False)
        self.assertIn("model_state_dict", ckpt)
        self.assertIn("e", ckpt)


if __name__ == "__main__":
    unittest.main()
