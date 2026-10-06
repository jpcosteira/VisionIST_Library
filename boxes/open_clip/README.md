# open_clip box (image + text embeddings, any OpenCLIP model)

A gRPC box around [open_clip](https://github.com/mlfoundations/open_clip). It
encodes images and/or text into one joint embedding space and returns the
embeddings, their cosine similarity and zero-shot probabilities.

It is the flexible sibling of the [`clip`](../clip) box: `clip` is fixed to
OpenAI's ViT-B/32 and returns torch tensors, while this one lets every request
choose any `(model, pretrained)` pair open_clip knows — LAION, DataComp, MetaCLIP,
SigLIP, EVA, a Hugging Face repo — and returns plain numpy arrays.

The box speaks the shared **envelope** interface:

```protobuf
service PipelineService { rpc Process( Envelope ) returns ( Envelope ); }
```

## Build and run

```bash
cd boxes/open_clip
docker build --tag sipgisr/visionist-open-clip --build-arg SERVICE_NAME=open_clip -f docker/Dockerfile .
docker run --rm --gpus all -p 8061:8061 -e PORT=8061 sipgisr/visionist-open-clip
```

The image bakes in the default checkpoint (`ViT-B-32` / `laion2b_s34b_b79k`,
about 600 MB) so the first call does not download anything. Build options:

| build arg | default | meaning |
|---|---|---|
| `OPEN_CLIP_MODEL` | `ViT-B-32` | model used when a request names none |
| `OPEN_CLIP_PRETRAINED` | `laion2b_s34b_b79k` | its checkpoint tag |
| `PRELOAD_WEIGHTS` | `true` | `false` builds a thin image; weights are then fetched on demand |

Any *other* model is downloaded from Hugging Face the first time a request asks
for it, so that call is slow and the container needs network access (or a
mounted `/workspace/.cache`). Runtime environment variables:

| variable | default | meaning |
|---|---|---|
| `OPEN_CLIP_MODEL`, `OPEN_CLIP_PRETRAINED` | as built | the default pair |
| `OPEN_CLIP_CACHE_SIZE` | `1` | models kept loaded; the least recently used is evicted |
| `OPEN_CLIP_IDLE_SECONDS` | `60` | idle time before GPU models move back to CPU (`0` disables) |

It runs on a CPU, only much more slowly.

### Big models

Any open_clip model works, including ViT-L-14, ViT-H-14, ViT-bigG-14, EVA02-E and
the SigLIP family; what limits you is memory. The box runs in **fp32**, so the
weights alone take roughly 4 bytes per parameter, on the GPU while a request runs
and in host RAM before and after (a model is always loaded on CPU first):

| model | parameters | weights in fp32 |
|---|---|---|
| ViT-B-32 | 151 M | 0.6 GB |
| ViT-L-14 | 428 M | 1.7 GB |
| ViT-H-14 | ~1.0 B | ~3.9 GB |
| ViT-bigG-14 | ~2.5 B | ~10 GB |

ViT-L-14 fits any recent GPU. ViT-H-14 wants roughly 8 GB or more, and ViT-bigG-14
or EVA02-E-14-plus a 24 GB card. `OPEN_CLIP_CACHE_SIZE=1` keeps only one model
resident, so alternating models reloads each time. The first request for a model
downloads it (ViT-L-14 about 1.7 GB), so it is slow. SigLIP models additionally
need `transformers` and `sentencepiece`, which are in `requirements.txt`.

## Service usage

### Commands

| command | does |
|---|---|
| `encode` (default) | embed `data.images` and/or `data.texts` |
| `models` | list the available `(model, pretrained)` pairs; `parameters.model` narrows it to one model |
| `reset` | unload every cached model and free the memory |

### Request (`encode`)

- `data["images"]` — list of image bytes (JPEG/PNG). Optional.
- `data["texts"]` — list of strings. Optional. At least one of the two is required.
- `config["open_clip"]["parameters"]`:

| parameter | default | meaning |
|---|---|---|
| `model` | `ViT-B-32` | architecture, or `hf-hub:<org>/<repo>`. `ViT-B/32` is accepted too |
| `pretrained` | `laion2b_s34b_b79k` | checkpoint tag, or a local checkpoint path. The default tag belongs to the default model; any other model must name its tag |
| `device` | `auto` | `auto`, `cuda` or `cpu` |
| `normalize` | `true` | L2-normalise the returned embeddings |
| `batch_size` | `32` | items per forward pass (memory only, not results) |

### Response

`config_json` carries `{open_clip: {status, model, pretrained, device, normalized,
num_images, num_texts, embedding_dim, logit_scale, runtime, encoding}}`. `data`
carries `np.save` blobs, declared with the `numpy` codec:

| field | shape | when | meaning |
|---|---|---|---|
| `image_emb` | `(num_images, D)` | images sent | float32 embeddings (`D` = 512 for ViT-B-32) |
| `text_emb` | `(num_texts, D)` | texts sent | float32, same space |
| `similarity` | `(num_images, num_texts)` | both sent | cosine similarity in [-1, 1] |
| `probs` | `(num_images, num_texts)` | both sent | softmax over the texts of `logit_scale * similarity`; rows sum to 1 |

`similarity` and `probs` are always computed from unit-length vectors, whatever
`normalize` says. Embeddings from different models or checkpoints are not
comparable with each other.

Errors come back as `status: "error"` with a message: an unknown model or tag
lists what *is* available, a model without a tag says so.

## Call with visionist_client

```python
from visionist_client import Visionist

b = Visionist("localhost:8061")

# what is available
print(b.run(config={"open_clip": {"command": "models",
                                  "parameters": {"model": "ViT-B-32"}}}).config)

# zero-shot classification with a larger model
res = b.run(
    data={"images": ["car.jpg"],
          "texts": ["a dog", "a cat", "a race car"]},
    config={"open_clip": {"command": "encode",
                          "parameters": {"model": "ViT-L-14",
                                         "pretrained": "datacomp_xl_s13b_b90k"}}},
)
print(res.probs)            # (2, 3), decoded with the declared numpy codec
```

## Tests

```bash
# no weights, no box: swaps in a randomly initialised model and checks everything around it
python test/smoke_inprocess.py

# against a running box; meaningful when it has real weights
python test/test_open_clip.py
OPEN_CLIP_TEST_MODEL=ViT-L-14 OPEN_CLIP_TEST_PRETRAINED=datacomp_xl_s13b_b90k python test/test_open_clip.py

# the shared contract test
python ../../contract/conformance/test_conformance.py --box open_clip --host localhost:8061
```

`smoke_inprocess.py` needs `torch`, `open_clip_torch`, `Pillow`, `numpy` and
`grpcio-tools` locally. It proves the envelope, validation, batching,
normalisation and cache behaviour; it cannot say anything about embedding
quality, which only a real checkpoint can.
