# -*- coding: utf-8 -*-
"""Cached, statically shaped decode for DECIMER 2.7.1.

The released SavedModel recomputes the whole token prefix at every step and
stores that prefix in a dynamic TensorArray. XLA has no kernel for
``TensorListResize``, so the official graph cannot be compiled. This module
rebuilds the same EfficientNetV2-M encoder and 6-layer decoder from the
DECIMER package, copies the released checkpoint by Keras variable name, and
runs one decode step with a fixed-length KV cache (query length 1, cache
padded to the model's maximum length). Image keys and values are computed
once. ``jit_compile=True`` fuses the step.

The encoder still applies EfficientNet preprocessing on top of
``config.decode_image``, which already applied it. That double preprocess is
what the SavedModel does; dropping it changes the tokens.

Positional encodings are added after the query and key projections, and the
value projection has none. Self-attention keys use the 1-d token positions;
cross-attention keys use the 16x16 image positions. Padding positions are
token id 0. Greedy decoding is fp32 argmax, then the package SELFIES decoder.

Requires the DECIMER 2.7.1 package (install with ``--no-deps``), TensorFlow
with CUDA, and ``tf-keras``. The SavedModel is the Zenodo release resolved
through pystow (``$PYSTOW_HOME/DECIMER-V2/DECIMER_model``), or ``--model-dir``.

XLA calls ``ptxas`` from ``PATH``. CUDA 12.4's assembler cannot target
compute capability 12.0; CUDA 12.8 or newer can. This module prepends a newer
``ptxas`` when it finds one, and sets ``--xla_gpu_autotune_level=0``. Left at
the default, the autotuner fails to compile the decoder's 512-wide matmuls.

    python tools/decimer_cached.py path/to/image.png
    python tools/decimer_cached.py --manifest path/to/manifest.jsonl --out predictions.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import site
import sys
import time
import types

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


def _prepend_ptxas():
    """Put a ptxas new enough for the GPU ahead of /usr/bin/ptxas."""
    candidates = []
    candidates.extend(sorted(glob.glob("/usr/local/cuda-*/bin"), reverse=True))
    candidates.append("/usr/local/cuda/bin")
    py = f"python{sys.version_info.major}.{sys.version_info.minor}"
    candidates.append(
        os.path.join(sys.prefix, "lib", py, "site-packages", "nvidia", "cuda_nvcc", "bin")
    )
    chosen = None
    for directory in candidates:
        if os.path.isfile(os.path.join(directory, "ptxas")):
            chosen = directory
            break
    if chosen is None:
        return None
    os.environ["PATH"] = chosen + os.pathsep + os.environ.get("PATH", "")
    return os.path.dirname(chosen)


_cuda_root = _prepend_ptxas()
_xla_flags = os.environ.get("XLA_FLAGS", "")
if "xla_gpu_autotune_level" not in _xla_flags:
    _xla_flags = (_xla_flags + " --xla_gpu_autotune_level=0").strip()
if _cuda_root and "xla_gpu_cuda_data_dir" not in _xla_flags:
    _xla_flags = (_xla_flags + f" --xla_gpu_cuda_data_dir={_cuda_root}").strip()
os.environ["XLA_FLAGS"] = _xla_flags

import tensorflow as tf
from tensorflow.core.protobuf import trackable_object_graph_pb2 as tog

N_LAYERS = 6
D_MODEL = 512
N_HEADS = 8
DEPTH = D_MODEL // N_HEADS
D_FF = 2048
IMG_TOKENS = 256


def configure():
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("no GPU visible; running on CPU", flush=True)
        return
    cap = os.environ.get("DECIMER_GPU_MEM_MB", "").strip()
    if cap:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=int(cap))],
        )
        # A virtual device is already set; a later set_memory_growth call errors.
        tf.config.experimental.set_memory_growth = lambda *args, **kwargs: None
    else:
        tf.config.experimental.set_memory_growth(gpus[0], True)


def install_decimer_stub():
    """Import DECIMER submodules without executing package ``__init__``.

    ``DECIMER/__init__.py`` imports ``decimer.py``, which loads both SavedModels
    at import time and will re-download them when the checkpoint size differs.
    """
    try:
        candidates = list(site.getsitepackages())
    except Exception:
        candidates = []
    candidates.extend(sys.path)
    pkg_dir = None
    for entry in candidates:
        cand = os.path.join(entry, "DECIMER")
        if os.path.isfile(os.path.join(cand, "Transformer_decoder.py")):
            pkg_dir = cand
            break
    if pkg_dir is None:
        raise SystemExit("DECIMER package not found on sys.path")
    pkg = types.ModuleType("DECIMER")
    pkg.__path__ = [pkg_dir]
    pkg.__package__ = "DECIMER"
    sys.modules["DECIMER"] = pkg


def shim_tokenizer():
    try:
        import keras.preprocessing.text  # noqa: F401
    except ImportError:
        import tf_keras.preprocessing.text as text

        sys.modules.setdefault(
            "keras.preprocessing", sys.modules["tf_keras.preprocessing"]
        )
        sys.modules["keras.preprocessing.text"] = text


def resolve_model_dir(explicit):
    if explicit:
        path = explicit
    else:
        home = os.environ.get("PYSTOW_HOME", "")
        if home:
            path = os.path.join(home, "DECIMER-V2", "DECIMER_model")
        else:
            # Optional: only needed when PYSTOW_HOME and --model-dir are both unset.
            import pystow

            path = os.path.join(str(pystow.join("DECIMER-V2")), "DECIMER_model")
    if not os.path.isfile(os.path.join(path, "saved_model.pb")):
        raise SystemExit(
            f"DECIMER SavedModel not found at {path}. "
            "Set PYSTOW_HOME or pass --model-dir."
        )
    return path


def load_assets(model_dir):
    shim_tokenizer()
    with open(os.path.join(model_dir, "assets", "tokenizer_SMILES.pkl"), "rb") as fh:
        tokenizer = pickle.load(fh)
    with open(os.path.join(model_dir, "assets", "max_length.pkl"), "rb") as fh:
        max_len = int(pickle.load(fh))
    return tokenizer, max_len


def load_keras_checkpoint(model_dir):
    """Map each Keras variable name to its checkpoint tensor.

    The SavedModel stores backbone weights as ``variables/N``. The object
    graph still records the Keras name (``eff_net_v2_model/blocks_0/...``)
    on that node, and the tensor lives at
    ``<node path>/.ATTRIBUTES/VARIABLE_VALUE``.
    """
    ckpt = os.path.join(model_dir, "variables", "variables")
    reader = tf.train.load_checkpoint(ckpt)
    valid = set(reader.get_variable_to_shape_map())
    raw = reader.get_tensor("_CHECKPOINTABLE_OBJECT_GRAPH")
    graph = tog.TrackableObjectGraph()
    graph.ParseFromString(bytes(raw))
    keras_to_key = {}

    def prefer(old, new):
        def score(key):
            if "/keras_api/" in key:
                return 3
            if "trainable_variables" in key or "non_trainable" in key:
                return 2
            return 0

        if old is None or score(new) < score(old):
            return new
        return old

    def walk(nid, path):
        node = graph.nodes[nid]
        names = [attr.full_name for attr in node.attributes if attr.full_name]
        if names:
            key = path + "/.ATTRIBUTES/VARIABLE_VALUE"
            if key in valid:
                for name in names:
                    keras_to_key[name] = prefer(keras_to_key.get(name), key)
        for child in node.children:
            child_path = f"{path}/{child.local_name}" if path else child.local_name
            walk(child.node_id, child_path)

    walk(0, "")
    return reader, keras_to_key


def assign_by_keras_name(reader, keras_to_key, model, label):
    missing = []
    copied = 0
    names = list(keras_to_key)
    for var in model.weights:
        name = var.name.split(":")[0]
        key = keras_to_key.get(name)
        if key is None:
            hits = [
                keras_to_key[candidate]
                for candidate in names
                if candidate.endswith("/" + name) or name.endswith("/" + candidate)
            ]
            key = hits[0] if len(hits) == 1 else None
        if key is None:
            missing.append(name)
            continue
        tensor = reader.get_tensor(key)
        if tuple(tensor.shape) != tuple(var.shape):
            missing.append(f"{name}: dst {tuple(var.shape)} src {tuple(tensor.shape)}")
            continue
        var.assign(tensor)
        copied += 1
    print(f"{label}: copied {copied}/{len(model.weights)}", flush=True)
    if missing or copied != len(model.weights):
        for line in missing[:12]:
            print(f"  missing {line}", flush=True)
        raise SystemExit(2)
    return copied


def build_models(config, encoder_mod, vocab_len, max_len):
    testing = config.Config()
    testing.initialize_encoder_config(
        image_embedding_dim=(IMG_TOKENS, D_MODEL),
        preprocessing_fn=tf.keras.applications.efficientnet.preprocess_input,
        backbone_fn=encoder_mod.get_efficientnetv2_backbone,
        image_shape=(512, 512, 3),
        do_permute=False,
    )
    testing.initialize_transformer_config(
        vocab_len=vocab_len,
        max_len=max_len,
        n_transformer_layers=N_LAYERS,
        transformer_d_dff=D_FF,
        transformer_n_heads=N_HEADS,
        image_embedding_dim=D_MODEL,
    )
    _optimizer, encoder, transformer = config.prepare_models(
        encoder_config=testing.encoder_config,
        transformer_config=testing.transformer_config,
        replica_batch_size=1,
        verbose=0,
    )
    transformer(
        tf.ones((1, 2), dtype=tf.int32),
        tf.zeros((1, IMG_TOKENS, D_MODEL)),
        training=False,
    )
    return encoder, transformer


def split_heads(x):
    seq = x.shape[1] if x.shape[1] is not None else tf.shape(x)[1]
    x = tf.reshape(x, [1, seq, N_HEADS, DEPTH])
    return tf.transpose(x, [0, 2, 1, 3])


def combine_heads(x):
    x = tf.transpose(x, [0, 2, 1, 3])
    seq = x.shape[1] if x.shape[1] is not None else tf.shape(x)[1]
    return tf.reshape(x, [1, seq, D_MODEL])


def attend(q, k, v, mask):
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    logits = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(dk)
    if mask is not None:
        logits = logits + mask * tf.constant(-1e9, tf.float32)
    weights = tf.nn.softmax(logits, axis=-1)
    return tf.matmul(weights, v)


def write_cache(cache, projected, index, max_len):
    """Insert one key/value of shape [1, heads, 1, depth] at ``index``."""
    token = tf.transpose(projected, [2, 0, 1, 3])
    hot = tf.one_hot(index, max_len)[:, None, None, None]
    return cache * (1.0 - hot) + token * hot


def make_step(transformer, max_len, jit_compile):
    pos_1d = transformer.pos_encoding_1d
    layers = list(transformer.dec_layers)
    sqrt_d = tf.math.sqrt(tf.constant(D_MODEL, tf.float32))

    def step(token, index, token_ids, self_k, self_v, cross_k, cross_v):
        hot = tf.one_hot(index, max_len)
        token_ids = tf.cast(
            tf.cast(token_ids, tf.float32) * (1.0 - hot)
            + tf.cast(token[0, 0], tf.float32) * hot,
            tf.int32,
        )
        x = transformer.embedding(token) * sqrt_d
        pos = tf.gather(pos_1d[0], index)[None, None, :]
        mask = tf.cast(tf.equal(token_ids, 0), tf.float32)[None, None, None, :]
        new_k = []
        new_v = []
        for i, layer in enumerate(layers):
            q = layer.mha1.wq(x) + pos
            k = layer.mha1.wk(x) + pos
            v = layer.mha1.wv(x)
            qh = split_heads(q)
            kh = split_heads(k)
            vh = split_heads(v)
            sk = write_cache(self_k[i], kh, index, max_len)
            sv = write_cache(self_v[i], vh, index, max_len)
            k_all = tf.transpose(sk, [1, 2, 0, 3])
            v_all = tf.transpose(sv, [1, 2, 0, 3])
            attn = combine_heads(attend(qh, k_all, v_all, mask))
            out1 = layer.layernorm1(layer.mha1.dense(attn) + x)

            q = layer.mha2.wq(out1) + pos
            qh = split_heads(q)
            attn = combine_heads(attend(qh, cross_k[i], cross_v[i], None))
            out2 = layer.layernorm2(layer.mha2.dense(attn) + out1)
            out3 = layer.layernorm3(layer.ffn(out2) + out2)
            x = out3
            new_k.append(sk)
            new_v.append(sv)
        probs = transformer.final_layer(x)
        return probs, token_ids, tf.stack(new_k), tf.stack(new_v)

    signature = [
        tf.TensorSpec([1, 1], tf.int32),
        tf.TensorSpec([], tf.int32),
        tf.TensorSpec([max_len], tf.int32),
        tf.TensorSpec([N_LAYERS, max_len, 1, N_HEADS, DEPTH], tf.float32),
        tf.TensorSpec([N_LAYERS, max_len, 1, N_HEADS, DEPTH], tf.float32),
        tf.TensorSpec([N_LAYERS, 1, N_HEADS, IMG_TOKENS, DEPTH], tf.float32),
        tf.TensorSpec([N_LAYERS, 1, N_HEADS, IMG_TOKENS, DEPTH], tf.float32),
    ]
    return tf.function(step, input_signature=signature, jit_compile=jit_compile)


def build_cross(transformer, enc_out):
    pos = transformer.pos_encoding_2d
    keys, values = [], []
    for layer in transformer.dec_layers:
        k = layer.mha2.wk(enc_out) + pos
        v = layer.mha2.wv(enc_out)
        keys.append(split_heads(k))
        values.append(split_heads(v))
    return tf.stack(keys), tf.stack(values)


def empty_cache(max_len):
    shape = [N_LAYERS, max_len, 1, N_HEADS, DEPTH]
    return (
        tf.zeros([max_len], tf.int32),
        tf.zeros(shape),
        tf.zeros(shape),
    )


def greedy(step_fn, start, end, cross_k, cross_v, max_len):
    token = tf.constant([[start]], tf.int32)
    token_ids, self_k, self_v = empty_cache(max_len)
    ids = []
    for t in range(max_len):
        probs, token_ids, self_k, self_v = step_fn(
            token,
            tf.constant(t, tf.int32),
            token_ids,
            self_k,
            self_v,
            cross_k,
            cross_v,
        )
        token = tf.cast(tf.argmax(probs, axis=-1), tf.int32)
        val = int(token.numpy()[0, 0])
        ids.append(val)
        if val == end:
            break
    return ids


def detokenize(tokenizer, utils, ids):
    words = [tokenizer.index_word[i] for i in ids]
    raw = "".join(words).replace("<start>", "").replace("<end>", "")
    return utils.decoder(raw)


def warmup(step_fn, max_len):
    """Compile the step once. A failed XLA call can poison later results."""
    token_ids, self_k, self_v = empty_cache(max_len)
    cross = tf.zeros([N_LAYERS, 1, N_HEADS, IMG_TOKENS, DEPTH])
    step_fn(
        tf.constant([[0]], tf.int32),
        tf.constant(0, tf.int32),
        token_ids,
        self_k,
        self_v,
        cross,
        cross,
    )


class Runner:
    def __init__(self, model_dir, use_xla):
        configure()
        install_decimer_stub()
        # After the stub. Importing DECIMER itself runs decimer.py and loads both SavedModels.
        import DECIMER.Efficient_Net_encoder as encoder_mod
        import DECIMER.config as config
        import DECIMER.utils as utils

        self.config = config
        self.utils = utils
        tokenizer, max_len = load_assets(model_dir)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.start = int(tokenizer.word_index["<start>"])
        self.end = int(tokenizer.word_index["<end>"])
        if self.start == 0:
            raise SystemExit("start token is 0; the pad mask would hide it")

        reader, keras_to_key = load_keras_checkpoint(model_dir)
        emb_key = keras_to_key.get("decoder/embedding/embeddings")
        if emb_key is None:
            raise SystemExit("checkpoint has no decoder/embedding/embeddings")
        vocab = int(reader.get_tensor(emb_key).shape[0])
        print(f"building encoder and decoder  vocab {vocab}  max_len {max_len}", flush=True)
        self.encoder, transformer = build_models(config, encoder_mod, vocab, max_len)
        assign_by_keras_name(reader, keras_to_key, self.encoder, "encoder")
        assign_by_keras_name(reader, keras_to_key, transformer, "decoder")
        self.transformer = transformer
        self.step_fn = make_step(transformer, max_len, jit_compile=use_xla)
        if use_xla:
            print("compiling the decode step", flush=True)
            try:
                warmup(self.step_fn, max_len)
            except Exception as exc:
                raise SystemExit(
                    "XLA compile failed. Install a ptxas that can target this GPU "
                    "(CUDA 12.8 or newer; /usr/bin/ptxas from CUDA 12.4 cannot "
                    f"assemble compute capability 12.0).\n{exc}"
                ) from exc
            print("XLA step ready", flush=True)

    def predict_path(self, path):
        image = self.config.decode_image(path)
        encoded = self.encoder(image[None, ...], training=False)
        cross_k, cross_v = build_cross(self.transformer, encoded)
        ids = greedy(
            self.step_fn, self.start, self.end, cross_k, cross_v, self.max_len
        )
        return detokenize(self.tokenizer, self.utils, ids)


def load_manifest(path, args):
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if args.template and row["template"] not in args.template.split(","):
                continue
            if args.variant and row["variant"] not in args.variant.split(","):
                continue
            if args.subset != "all" and row["subset"] != args.subset:
                continue
            if args.ids and row["compound_id"] not in args.ids.split(","):
                continue
            rows.append(row)
    return rows


def image_path(images_root, rel):
    return os.path.join(images_root, *rel.split("/"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="*", help="image files to decode; prints one SMILES per line")
    ap.add_argument("--manifest", default="", help="benchmark manifest JSONL")
    ap.add_argument("--out", default="predictions/decimer_cached.jsonl")
    ap.add_argument("--images-root", default="", help="root the manifest image_path is relative to")
    ap.add_argument("--model-dir", default="", help="DECIMER SavedModel directory")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--template", default="")
    ap.add_argument("--variant", default="")
    ap.add_argument("--subset", default="all", choices=["all", "organic", "coordination"])
    ap.add_argument("--ids", default="")
    ap.add_argument(
        "--no-xla",
        action="store_true",
        help="run the static graph step without jit_compile (about 12 ms/token)",
    )
    args = ap.parse_args()
    if bool(args.manifest) == bool(args.images):
        ap.error("pass image paths or --manifest, not both")

    model_dir = resolve_model_dir(args.model_dir or None)
    runner = Runner(model_dir, use_xla=not args.no_xla)

    if args.images:
        for path in args.images:
            print(runner.predict_path(path), flush=True)
        return

    if not args.images_root:
        args.images_root = os.path.dirname(os.path.dirname(os.path.abspath(args.manifest)))
    rows = load_manifest(args.manifest, args)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    done = set()
    if os.path.exists(args.out):
        with open(args.out, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["image_id"])
                except Exception:
                    pass
    todo = [row for row in rows if row["image_id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"manifest rows {len(rows)}, already done {len(done)}, to run {len(todo)}", flush=True)
    if not todo:
        print("nothing to do")
        return

    t0 = time.time()
    n = 0
    with open(args.out, "a", encoding="utf-8") as out:
        for row in todo:
            path = image_path(args.images_root, row["image_path"])
            try:
                smiles = runner.predict_path(path) or ""
            except Exception as exc:
                print(f"  skip {row['image_id']} {exc!r}"[:300], flush=True)
                smiles = ""
            out.write(json.dumps({
                "image_id": row["image_id"],
                "image_path": row["image_path"],
                "prediction": str(smiles).strip(),
            }, ensure_ascii=False) + "\n")
            out.flush()
            n += 1
            if n == 1 or n % 20 == 0 or n == len(todo):
                elapsed = time.time() - t0
                print(f"[{n}/{len(todo)}] {elapsed:.0f}s  {elapsed / n:.2f}s/img", flush=True)
    print(f"done: {n} predictions in {time.time() - t0:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
