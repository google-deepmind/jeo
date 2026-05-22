# Preprocessing Operations in Jeo

This page lists the available preprocessing operations in `jeo/pp` that can
be used to construct preprocessing pipelines, as well as details on the
mini-language used to specify them.

## Preprocessing Spec Construction & Grammar

The preprocessing pipeline is described by a string mini-language, where
operations are separated by the pipe (`|`) character. The grammar is as
follows:

```
fn1|fn2(arg1, arg2, ...)|fn3(kwarg1=val1, kwarg2=val2)|...
```

- Operations are applied sequentially to the data dictionary.
- Each operation can optionally have one or more arguments (positional or keyword).
- Arguments are parsed using Python's `ast.literal_eval`, so you can use
  Python literals like strings, numbers, lists, tuples, and dicts.
- **Parentheses can be dropped** if no arguments are given (e.g., `decode|flip_lr`).
- The output of the preprocessing function is always a dictionary of tensors.

Example:
```python
config.pp_train = "decode|resize(224)|value_range(-1, 1)|keep('image', 'labels')"
```

### Loading Operations

By default, Jeo automatically loads preprocessing ops from [`jeo/pp/`](https://github.com/google-deepmind/jeo/tree/main/jeo/pp/)
(specifically imported in [`jeo/input_pipeline.py`](https://github.com/google-deepmind/jeo/tree/main/jeo/input_pipeline.py)).

If you need to use ops from other modules (e.g., project-specific ops or
big_vision ops), you must include them in the config's `pp_module` or
`pp_modules` list:

```python
config.pp_modules = [
    "proj.forty.pp_ops",  # Loads from jeo/proj/forty/pp_ops.py
    "proj.forestcast.pp_ops",  # Loads from jeo/proj/forestcast/pp_ops.py
]
```

### Decorators and Custom Ops

Many operations use the `@utils.InKeyOutKey` decorator.

- This decorator automatically extracts the feature specified by `key`,
  `inkey`, or `outkey` (defaulting to `"image"`).
- The decorated function receives this feature directly (instead of the
  whole dictionary) and returns the modified feature, which is then
  automatically updated in the data dictionary.

### Deterministic Random Operations

For multi-modal or multi-temporal data, you often want to apply the *same*
random transformation to multiple features (e.g., rotating both the image
and the segmentation mask by the same random angle).
Jeo follows the approach from the Big Vision UViM project:

1.  **Generate a 'named' random value feature first** using `randu` or `randint`.
2.  **Apply deterministic operations** based on that random feature across
    the features of interest.

Examples:

- Random flip and rotation applied to both 'image' and 'mask':

  `"randu('flip_rot')|det_flip_rot(key='image')|det_flip_rot(key='mask')"`
- Random crop:

  `"randu('crop')|det_crop(128,key='image')|det_crop(128,key='mask')"`

Note: Check the specific op implementation in [`pp/rand_det_ops.py`](https://github.com/google-deepmind/jeo/tree/main/jeo/pp/rand_det_ops.py) to see
the default expected name of the random value key (e.g., `"crop"`, `"flip_rot"`,
etc.).

## Best Practices

- **Always end with `keep(...)`**: It is highly recommended to end your
  preprocessing pipeline with a `keep(...)` operation to drop all unused
  keys. Leftover keys with incompatible types (like strings or ragged
  tensors) can cause dtype errors in JAX during training.

---

## Available Operations

### Basic Operations ([pp_ops.py](https://github.com/google-deepmind/jeo/tree/main/jeo/pp/pp_ops.py))

| Op Name | Arguments | Description |
| :--- | :--- | :--- |
| `add_meta_as_channel` | `meta_key: str, to_key: str = "image"` | Appends scalar value as an additional channel. |
| `cast` | `key: str, dtype: str = "int32"` | Casts a given feature to a given type. |
| `check_finite` | `keys: Sequence[str] \| None = None, msg: str = ""` | Checks for NaNs and infs in given keys. |
| `debug_assert_shapes` | `shapes: dict[str, Sequence[int]]` | Assert shapes for given keys. |
| `ensure_4d` | `keys: Sequence[str] \| str, extra_axis: int = 0` | Ensures number of dimension for given key field is 4. |
| `ensure_shape` | `shape: Sequence[int]` | Ensures shape for given key field (for debugging). |
| `expand_dim` | `*keys: Sequence[str], axis: int = 0` | Expands by one dimension at given axis. |
| `extract_channels` | `from_key: str, key_channels: dict, axis=-1` | Extracts channels to separate tensors. |
| `invert_bool` | `keys: str \| Sequence[str]` | Inverts boolean input values. |
| `jeo_concat` | `keys: Sequence[str], outkey="image", axis=-1, pop_origs=True, broadcast=False` | Concatenates images across a given axis. |
| `jeo_drop` | `keys: Sequence[str]` | Drops the given keys. |
| `jeo_keep` | `keys: Sequence[str]` | Keeps only the given keys. |
| `max` | `*keys: Sequence[str], axis: int = 0, keepdims: bool = False` | Runs reduce_max on a given axis. |
| `reduce` | `keys: Sequence[str], outkey: str, reduction: str = "prod"` | Combines multiple features into one with a specified reduction. |
| `remap` | `mapping: dict, default_value: Any = 0` | Remapping values from a feature. |
| `remap_ints` | `mapping: list[int]` | Mapping from a set of integer labels to another set. |
| `rename` | `**kv_pairs: dict[str, str]` | Renames selected keys. |
| `select_channels` | `key_channels: int \| Sequence[int] \| dict, axis: int = -1` | Selects channels by index. |
| `select_channels_by_name` | `key: str, channels: Sequence[str], names: Sequence[str], axis=-1` | Selects channels by name. |
| `skai_concat` | `key1="pre_image_png", key2="post_image_png", outkey="image", axis=-1, pop_origs=True` | Concatenates images across a given axis. |
| `squeeze_dim` | `*keys: Sequence[str], axis: int = 0` | Squeezes by one dimension at given axis. |
| `string_to_int` | `key: str` | Converts string key to int. |
| `transpose` | `key: str, perm: Sequence[int]` | Transposes a given feature. |

### Big Vision Forked Operations ([bv_ops.py](https://github.com/google-deepmind/jeo/tree/main/jeo/pp/bv_ops.py))

These ops are forked from the big_vision codebase.

| Op Name | Arguments | Description |
| :--- | :--- | :--- |
| `choice` | `n: int \| str = "single", inkey=None, outkey=None, key=None` | Chooses the same `n` random entries of all `keys`. |
| `concat` | `inkeys: Sequence[str], outkey: str \| None = None, axis=-1` | Concatenates elements along some axis. |
| `copy` | `inkey: str, outkey: str` | Copies value of `inkey` into `outkey`. |
| `drop` | `*keys: Sequence[str]` | Drops the given keys. |
| `flatten` | None | Flattens the keys of data with separator `/`. |
| `keep` | `*keys: Sequence[str]` | Keeps only the given keys. |
| `lookup` | `mapping: str, npzkey="fnames", sep: str \| None = None` | Map string to number using a file. |
| `onehot` | `depth: int, key="labels", key_result=None, multi=True, on=1.0, off=0.0` | One-hot encodes the input. |
| `pad_to_shape` | `shape: Sequence[int], pad_value: int = 0, where: str = "after"` | Pads tensor to specified shape. |
| `rag_tensor` | None | Converts the specified feature to ragged tensor. |
| `reshape` | `new_shape: Sequence[int]` | Reshapes tensor to a given new shape. |
| `setdefault` | `key: str, value: Any` | If `key` is an empty tensor, set it to `value`. |
| `squeeze_last_dim` | None | Squeezes the last dimension. |
| `value_range` | `vmin=-1, vmax=1, in_min=0, in_max=255.0, clip_values=False` | Transforms an image to `[vmin, vmax]` range. |

### Deterministic Random Operations ([rand_det_ops.py](https://github.com/google-deepmind/jeo/tree/main/jeo/pp/rand_det_ops.py))

These operations use a provided random key (usually in range `[0, 1]`) to
perform operations deterministically.

| Op Name | Arguments | Description |
| :--- | :--- | :--- |
| `det_crop` | `crop_size: int, num_crops=1, randkey="crop"` | Deterministically crops an image. |
| `det_flip_rot` | `randkey="flip_rot"` | Deterministically flip and rotate by orthogonal angle. |
| `det_fliplr` | `randkey="fliplr"` | Flips an image horizontally based on `randkey`. |
| `det_noise` | `noise_min=0.0, noise_max=1.0, multiplicative=True, randkey="noise", pixel_wise=True` | Deterministically insert noise. |
| `det_resize` | `min_ratio=0.8, max_ratio=1.3, randkey="resize"` | Deterministically resizes an image. |
| `det_roll` | `randkey1="rollx", randkey2="rolly"` | Deterministically rolls image. |
| `det_rotate` | `randkey="angle", interpolation="bilinear", fill_value=0` | Rotate by an angle using `randkey`. |
| `det_rotate90` | `randkey="angle"` | Rotate by an orthogonal angle. |
| `det_select_channels` | `randkey="start_index_key", num_channels=1, axis=0` | Deterministically selects random channels. |
| `randint` | `key: str, minval=0, maxval=1<<31, step=1` | Creates a random uniform integer in `key`. |
| `randu` | `key: str` | Creates a random uniform float `[0, 1)` in `key`. |

### Satellite Specific Operations ([sat_ops.py](https://github.com/google-deepmind/jeo/tree/main/jeo/pp/sat_ops.py))

| Op Name | Arguments | Description |
| :--- | :--- | :--- |
| `add_lat_lon_channels` | `key: str` | Appends unit-sphere cartesian coordinates as channels. |
| `concat_satellites` | `sats: tuple[str, ...], key: str \| None = None` | Concatenates multiple satellite inputs. |
| `merge_all_dimensions` | `keys: tuple = ("image",)` | Merges all dimensions into a single dimension. |
| `merge_spatial_dimensions` | `keys: tuple = ("image",)` | Merges spatial dimensions into a single dimension. |
| `pad_seq` | `keys: str \| list[str], length: int, allow_crop: bool = True` | Pads (and crops) sequences in first dimension. |
| `pad_tensors` | `keys: str \| list[str], dims_to_pad: list[int]` | Pads tensors to match the max dim's length. |
| `rearrange` | `pattern: str, keys: tuple = ("image",), **kwargs` | Arbitrary rearranges dimensions using einops. |
| `reduce_temporal` | `keys: tuple = ("image",), kind="first", timestamp_key=None, repeat=None` | Reduces temporal dimension to a single element. |
| `s1_to_rgb` | `temporal="first", bands=(0, 1), vmin=-25.0, vmax=0.0` | Gather Sentinel-1 bands and convert to pseudo-RGB. |
| `s2_to_rgb` | `bands=(2, 1, 0), temporal="first", bias=0, scale=3000` | Gather Sentinel-2 bands and convert to RGB. |
| `scale_sentinel2_by_tanh` | `key: str, imagery_type: str` | Rescale Sentinel-2 imagery with tanh normalization. |
| `scale_spatial_dims_like` | `keys_to_resize, reference_key, height_axis=-2, width_axis=-3` | Scales spatial dims to match reference key. |

### Image Operations ([image_ops.py](https://github.com/google-deepmind/jeo/tree/main/jeo/pp/image_ops.py))

| Op Name | Arguments | Description |
| :--- | :--- | :--- |
| `central_crop` | `crop_size: int \| None = None` | Makes central crop of a given size. |
| `clear_boundary` | `margin: int, inverse=False` | Sets a zero value on a boundary of a tensor. |
| `cutout_from_mask` | `mask_key="cutout_mask", replace=0` | Cuts out a region according to mask. |
| `cutout_mask` | `min_size_ratio=0.01, max_size_ratio=0.5, inkey="image", outkey="cutout_mask"` | Creates a mask to cutout a region. |
| `decode` | `channels=3, precise=False` | Decode an encoded image string. |
| `decode_jpeg_and_inception_crop` | `size=None, area_min=5, ...` | Decode jpeg string and make inception crop. |
| `extract_patches` | `crop_size: int, stride=1, flatten=False` | Extract overlapping patches from an image. |
| `flip_lr` | None | Flips an image horizontally with probability 50%. |
| `flip_ud` | None | Flips an image vertically with probability 50%. |
| `flip_ud_with_label` | `outkey="flipped", key="image"` | Flips image up/down and saves if it was flipped. |
| `inception_crop` | `size=None, area_min=5, area_max=100, method="bilinear", antialias=False` | Makes inception-style image crop. |
| `pansharpen` | `keys: str \| Sequence[str], out_key="image"` | Pansharpen RGB image with Pan channel. |
| `rand_shift` | `max_shift=4` | Randomly shifts given image with reflection. |
| `random_crop` | `crop_size: int, num_crops=1` | Makes a random crop of a given size. |
| `random_fill_along_dim` | `axis=1, probability=0.25, fill_value=0` | Random masks some channels from a given dimension. |
| `random_resize` | `ref_size: int, min_ratio: float, max_ratio: float` | Randomly resizes image. |
| `resize` | `size: int \| Sequence[int], method="bilinear", antialias=False` | Resizes image to a given size. |
| `resize_long` | `longer_size: int, method="area", antialias=True` | Resizes longer side preserving aspect ratio. |
| `resize_small` | `smaller_size: int, method="area", antialias=False` | Resizes smaller side preserving aspect ratio. |
| `rot90` | None | Randomly rotate an image by multiples of 90 degrees. |
| `tanh_value_range` | `mean, std, tanh_scale=1.0` | Normalizing image pixels by a scaled tanh function. |
| `to_grayscale` | `keep_dims=True` | Converts RGB images to grayscale. |
| `vgg_value_range` | `mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), scale=255.0` | VGG-style preprocessing. |

### Internal & Specialized Operations ([internal_ops.py](https://github.com/google-deepmind/jeo/tree/main/jeo/pp/internal_ops.py), [decode_ops.py](https://github.com/google-deepmind/jeo/tree/main/jeo/pp/decode_ops.py))

| Op Name | Arguments | Description | File |
| :--- | :--- | :--- | :--- |
| `cutout` | `size=20, replace=128` | Cuts out a square of fixed size. | [`internal_ops.py`](https://github.com/google-deepmind/jeo/tree/main/jeo/pp/internal_ops.py) |
| `cutout_randsize` | `min_size_ratio=0.1, max_size_ratio=0.5, replace=128` | Cuts out a random-sized box. | [`internal_ops.py`](https://github.com/google-deepmind/jeo/tree/main/jeo/pp/internal_ops.py) |
| `decode_jeo_satellites` | `sat_keys, data_dir, dataset, ...` | Decodes standardized drivers example. | [`decode_ops.py`](https://github.com/google-deepmind/jeo/tree/main/jeo/pp/decode_ops.py) |
| `randaug_geo` | `num_layers=2, magnitude=10, ...` | Applies RandAugment for geospatial data. | [`internal_ops.py`](https://github.com/google-deepmind/jeo/tree/main/jeo/pp/internal_ops.py) |
| `simclr_crop_and_resize` | `height, width=None` | Random crop and resize as in SimCLR. | [`internal_ops.py`](https://github.com/google-deepmind/jeo/tree/main/jeo/pp/internal_ops.py) |
| `simclr_random_color_jitter` | `p=1.0, strength=0.5` | Color jittering as in SimCLR. | [`internal_ops.py`](https://github.com/google-deepmind/jeo/tree/main/jeo/pp/internal_ops.py) |
