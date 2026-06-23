import os
import random
from typing import Any, Dict, List, Optional

from einops import rearrange
import numpy as np
from pydantic import Field, PrivateAttr
import torch
from transformers import AutoProcessor, ProcessorMixin, AutoTokenizer
from transformers.data.data_collator import DataCollatorMixin
from transformers.feature_extraction_utils import BatchFeature
import tree
import re
import ftfy
import html
import regex as re
import ast

from groot.vla.data.schema import (
    EmbodimentTag,
    DatasetMetadata,
)
from groot.vla.data.transform.base import InvertibleModalityTransform
from groot.vla.model.dreamzero.transform.common import formalize_language


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()

def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


class HuggingfaceTokenizer:

    def __init__(self, name, seq_len=None, clean=None, **kwargs):
        assert clean in (None, 'whitespace')
        self.name = name
        self.seq_len = seq_len
        self.clean = clean

        # When loading from a local checkpoint path (e.g. from training runs), pass
        # local_files_only=True to avoid HFValidationError from validate_repo_id.
        load_kwargs = dict(kwargs)
        if os.path.isdir(name):
            load_kwargs.setdefault("local_files_only", True)
        # init tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(name, **load_kwargs)
        self.vocab_size = self.tokenizer.vocab_size

    def __call__(self, sequence, **kwargs):
        return_mask = kwargs.pop('return_mask', False)

        # arguments
        _kwargs = {'return_tensors': 'pt'}
        if self.seq_len is not None:
            _kwargs.update({
                'padding': 'max_length',
                'truncation': True,
                'max_length': self.seq_len
            })
        _kwargs.update(**kwargs)


        # tokenization
        if isinstance(sequence, str):
            sequence = [sequence]
        if self.clean:
            sequence = [self._clean(u) for u in sequence]
        ids = self.tokenizer(sequence, **_kwargs)

        # output
        if return_mask:
            return ids.input_ids, ids.attention_mask
        else:
            return ids.input_ids

    def _clean(self, text):
        if self.clean == 'whitespace':
            text = whitespace_clean(basic_clean(text))
        # elif self.clean == 'lower':
        #     text = whitespace_clean(basic_clean(text)).lower()
        # elif self.clean == 'canonicalize':
        #     text = canonicalize(basic_clean(text))
        return text


def _embodiment_id(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.item())
    if isinstance(value, np.ndarray):
        return int(value.item())
    return int(value)


def _mapping_value(embodiment_tag_mapping: dict[str, int], tag: EmbodimentTag) -> int | None:
    if embodiment_tag_mapping is None:
        return None
    return embodiment_tag_mapping.get(tag.value)


def _tag_matches(value: Any, tag: EmbodimentTag) -> bool:
    if value is None:
        return False
    if isinstance(value, EmbodimentTag):
        return value == tag
    return str(value) == tag.value


def normalize_language_item(item: Any) -> str:
    """Match the historical collator's language literal handling."""
    try:
        parsed_item = ast.literal_eval(item)
        if isinstance(parsed_item, (list, tuple)):
            return str(parsed_item[0])
        return str(parsed_item)
    except (ValueError, SyntaxError, TypeError):
        return str(item)


def split_language_variants(item: Any) -> list[str]:
    """Return non-empty ``@``-separated language variants after normalisation.

    Some datasets store prompt augmentation in a single task string such as
    ``"Pick up the cup.@Lift the cup."``.  Training should sample one variant,
    while offline T5 cache precompute must materialize every variant.
    """
    text = normalize_language_item(item)
    variants = [variant.strip() for variant in text.split("@")]
    variants = [variant for variant in variants if variant]
    return variants or [text]


def select_language_variant(item: Any, *, training: bool) -> str:
    """Select one language variant for a training/eval sample."""
    variants = split_language_variants(item)
    if training and len(variants) > 1:
        return random.choice(variants)
    return variants[0]


def format_dreamzero_prompt(
    item: Any,
    *,
    embodiment_id: int,
    num_views: int,
    embodiment_tag_mapping: dict[str, int],
    embodiment_tag: EmbodimentTag | str | None = None,
) -> str:
    """Build the exact prompt string used before T5 encoding.

    This is shared by the training collator and the offline T5 precompute
    script so cache keys are derived from the same final text that the model
    would otherwise encode online.
    """
    processed_item = normalize_language_item(item)
    lower_item = processed_item.lower()
    embodiment_id = _embodiment_id(embodiment_id)

    if _tag_matches(embodiment_tag, EmbodimentTag.UNITREE_G1_UPPER_BODY) and num_views <= 1:
        return (
            "A single view video shows that a robot "
            + lower_item
            + " The view shows the camera view from the robot's head. The robot "
            + lower_item
        )
    if num_views > 1 and embodiment_id == _mapping_value(embodiment_tag_mapping, EmbodimentTag.AGIBOT):
        return (
            "A multi-view video shows that a robot "
            + lower_item
            + " The video is split into four views: The top-left view shows the camera view from the robot's head, the top-right view shows the camera view from the right hand, the bottom-left view shows the camera view from the left hand, and the bottom-right view is a black screen (inactive view). The robot "
            + lower_item
        )
    if num_views <= 1 and embodiment_id == _mapping_value(embodiment_tag_mapping, EmbodimentTag.UNITREE_G1_UPPER_BODY):
        return (
            "A single view video shows that a robot "
            + lower_item
            + " The view shows the camera view from the robot's head. The robot "
            + lower_item
        )
    if embodiment_id == _mapping_value(embodiment_tag_mapping, EmbodimentTag.OXE_DROID):
        return (
            "A multi-view video shows that a robot "
            + lower_item
            + " The video is split into three views: The top view shows the camera view from the robot's wrist, the bottom-left view shows the camera view from the left exterior camera, and the bottom-right view shows the camera view from the right exterior camera. During training, one of the two bottom exterior views may be a black screen (dropped view). The robot "
            + lower_item
        )
    if num_views > 1 and embodiment_id == _mapping_value(embodiment_tag_mapping, EmbodimentTag.LIBERO_SIM):
        return (
            "A multi-view video shows that a robot "
            + lower_item
            + " The video is split into two equal side-by-side views: the left half shows the agentview camera, and the right half shows the robot's eye-in-hand wrist camera. The robot "
            + lower_item
        )
    if embodiment_id == _mapping_value(embodiment_tag_mapping, EmbodimentTag.GR1_UNIFIED):
        return "A single view video shows that a human " + lower_item
    if embodiment_id == _mapping_value(embodiment_tag_mapping, EmbodimentTag.MECKA_HANDS):
        return "A single view video shows that a human " + lower_item
    if embodiment_id == _mapping_value(embodiment_tag_mapping, EmbodimentTag.XDOF):
        return (
            "A multi-view video shows that a robot "
            + lower_item
            + " The video is split into four views: The top-left view shows the camera view from the robot's head, the top-right view shows the camera view from the right hand, the bottom-left view shows the camera view from the left hand, and the bottom-right view is a black screen (inactive view). The robot "
            + lower_item
        )
    if embodiment_id == _mapping_value(embodiment_tag_mapping, EmbodimentTag.YAM):
        return (
            "A multi-view video shows that a robot "
            + lower_item
            + " The video is split into four views: The top-left view shows the top camera, the top-right view shows the right camera, the bottom-left view shows the left camera, and the bottom-right view is a black screen. The robot "
            + lower_item
        )
    raise ValueError(f"Embodiment ID {embodiment_id} not supported.")


TEXT_CONTEXT_KEY = "text_context"
TEXT_CONTEXT_MASK_KEY = "text_context_mask"
TEXT_CONTEXT_CACHE_PATH_KEY = "text_context_cache_path"


def text_embedding_cache_path(
    cache_dir: str,
    prompt: str,
    *,
    max_length: int,
    cache_tag: str,
) -> str:
    import hashlib

    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return os.path.join(cache_dir, f"{prompt_hash}.t5_len{max_length}.{cache_tag}.pt")


def collate(features: List[dict], tokenizer: AutoTokenizer, num_views=3, embodiment_tag_mapping=None) -> dict:
    batch = {}
    keys = features[0].keys()

    for key in keys:
        if key == "text":
            output_values = []
            for elem in features:
                output_values.append(
                    format_dreamzero_prompt(
                        elem[key],
                        embodiment_id=elem["embodiment_id"],
                        num_views=num_views,
                        embodiment_tag_mapping=embodiment_tag_mapping,
                    )
                )
            # print("output_values", output_values)
            ids, mask = tokenizer(output_values, return_mask=True, add_special_tokens=True)
            batch[key] = ids 
            batch['text_attention_mask'] = mask
        elif key == "text_negative":
            values = [elem[key] for elem in features]
            ids, mask = tokenizer(values, return_mask=True, add_special_tokens=True)
            batch[key] = ids 
            batch['text_attention_mask_negative'] = mask
        elif key in {TEXT_CONTEXT_KEY, TEXT_CONTEXT_MASK_KEY}:
            values = [elem[key] for elem in features]
            batch[key] = torch.stack([torch.as_tensor(value) for value in values])
        elif key == TEXT_CONTEXT_CACHE_PATH_KEY:
            batch[key] = [str(elem[key]) for elem in features]
        else:
            values = [elem[key] for elem in features]
            batch[key] = torch.from_numpy(np.stack(values))
    return batch



class DefaultDataCollator(DataCollatorMixin):
    def __init__(self, tokenizer_path: str="google/umt5-xxl", max_length: int=512, num_views: int=1, embodiment_tag_mapping=None):
        super().__init__()
        self.tokenizer = HuggingfaceTokenizer(name=tokenizer_path, seq_len=max_length, clean='whitespace')
        self.num_views = num_views
        self.embodiment_tag_mapping = embodiment_tag_mapping

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        return collate(features, self.tokenizer, self.num_views, self.embodiment_tag_mapping)


class DreamTransform(InvertibleModalityTransform):

    # -- We inherit from ModalityTransform, so we keep apply_to as well --
    apply_to: list[str] = Field(
        default_factory=list, description="Not used in this transform, kept for compatibility."
    )
    training: bool = Field(
        default=True, description="Whether to apply the transform in training mode."
    )

    formalize_language: bool = Field(default=False, description="Formalize language if True.")

    embodiment_tag_mapping: dict[str, int] = Field(
        default_factory=dict,
        description="The projector index of each embodiment tag.",
    )

    language_dropout_prob: float = Field(
        default=0.0,
        description="Dropout probability for language.",
    )
    always_use_default_instruction: bool = Field(
        default=False,
        description="Whether to always use the default instruction. For studying how much the language helps.",
    )

    # Private attributes to keep track of shapes/dimensions across apply/unapply
    _language_key: Optional[str] = PrivateAttr(default=None)
    _language_keys: Optional[list[str]] = PrivateAttr(default=None)

    # XEmbDiT arguments
    default_instruction: str
    max_state_dim: int
    max_action_dim: int
    max_length: int = 512
    embodiment_tag: EmbodimentTag | None = None
    state_horizon: int
    action_horizon: int
    num_views: int = 3

    # Add tokenizer attribute
    tokenizer_path: str = Field(
        default="google/umt5-xxl",
        description="Path to the tokenizer."
    )
    text_embedding_cache_dir: Optional[str] = Field(
        default=None,
        description="Directory containing offline T5 embedding cache files keyed by final DreamZero prompt.",
    )
    text_embedding_cache_tag: str = Field(
        default="dreamzero_wan_t5",
        description="Cache filename tag identifying the T5 encoder/checkpoint used for precompute.",
    )
    require_text_embedding_cache: bool = Field(
        default=False,
        description="If True, fail when text_embedding_cache_dir is unset or a prompt cache is missing.",
    )
    text_embedding_cache_runtime: str = Field(
        default="model",
        description=(
            "Where cached text embeddings are materialized: 'model' passes cache paths "
            "to the action head for per-rank GPU memoization; 'dataset' preserves the "
            "legacy behavior of loading tensors in the data transform."
        ),
    )
    _tokenizer: Optional[HuggingfaceTokenizer] = PrivateAttr(default=None)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize the tokenizer
        self._tokenizer = HuggingfaceTokenizer(
            name=self.tokenizer_path, 
            seq_len=self.max_length, 
            clean='whitespace'
        )
    
    @property
    def tokenizer(self):
        return self._tokenizer

    def _text_embedding_cache_path_for_prompt(self, prompt: str) -> str | None:
        if self.text_embedding_cache_dir is None:
            if self.require_text_embedding_cache:
                raise ValueError(
                    "require_text_embedding_cache=true but text_embedding_cache_dir is not set."
                )
            return None

        cache_path = text_embedding_cache_path(
            self.text_embedding_cache_dir,
            prompt,
            max_length=self.max_length,
            cache_tag=self.text_embedding_cache_tag,
        )
        if not os.path.exists(cache_path):
            prompt_preview = prompt.replace("\n", "\\n")
            if len(prompt_preview) > 1200:
                prompt_preview = prompt_preview[:1200] + "...<truncated>"
            raise FileNotFoundError(
                f"Missing DreamZero text embedding cache: {cache_path}. "
                "Run scripts/data/precompute_t5_text_embeddings.py with the same prompt/config first. "
                f"prompt_preview={prompt_preview!r}"
            )
        return cache_path

    def _load_cached_text_context(self, cache_path: str) -> tuple[torch.Tensor, torch.Tensor]:
        payload = torch.load(cache_path, map_location="cpu")
        context = payload["context"]
        context_mask = payload["mask"].bool()
        if context.ndim != 2:
            raise ValueError(f"Cached context must be [L, D], got {tuple(context.shape)} in {cache_path}")
        if context_mask.ndim != 1:
            raise ValueError(f"Cached mask must be [L], got {tuple(context_mask.shape)} in {cache_path}")
        if context.shape[0] != self.max_length or context_mask.shape[0] != self.max_length:
            raise ValueError(
                f"Cached context length mismatch for {cache_path}: "
                f"context={tuple(context.shape)}, mask={tuple(context_mask.shape)}, expected L={self.max_length}"
            )
        context = context.clone()
        context[~context_mask] = 0
        return context, context_mask

    def set_metadata(
        self, dataset_metadata: DatasetMetadata
    ):
        self.embodiment_tag = dataset_metadata.embodiment_tag

    def get_embodiment_tag(self) -> int:
        """Get the embodiment tag from the data."""
        assert (
            self.embodiment_tag is not None
        ), "Embodiment tag not set. Please call set_metadata first."
        return self.embodiment_tag_mapping[self.embodiment_tag.value]

    def check_keys_and_batch_size(self, data):
        grouped_keys = {}
        for key in data.keys():
            try:
                modality, _ = key.split(".")
                if "annotation" in key:
                    modality = "language"
            except:  # noqa: E722
                ### Handle language annotation special case
                if "annotation" in key:
                    modality = "language"
                else:
                    modality = "others"  # will contain the video, state, and action
            if modality not in grouped_keys:
                grouped_keys[modality] = []
            grouped_keys[modality].append(key)
        # Use video key to determine batch size.
        video_ndim = data["video"].ndim
        if video_ndim == 5:  # Interpret as [T, V, H, W, C]
            is_batched = False
            batch_size = 1
        elif video_ndim == 6:  # Interpret as [B, T, V, H, W, C]
            is_batched = True
            batch_size = data["video"].shape[0]
        else:
            raise ValueError(f"Unsupported video number of dimensions: {video_ndim}")

        # Handle language
        if "language" in grouped_keys:
            language_keys = grouped_keys["language"]
            self._language_keys = language_keys  # Store all keys for random selection
            if len(language_keys) == 1:
                self._language_key = language_keys[0]
            else:
                self._language_key = None  # Will be selected randomly in _prepare_language
        return is_batched, batch_size

    def _apply_vlm_processing(self, batch: dict) -> BatchFeature:
        """
        Args:
            batch:
                video: [V, T, C, H, W]
        Returns: required input with the format `BatchFeature`
        """
        images = batch["images"]  # [V, T, C, H, W]

        np_images = rearrange(images, "v t c h w -> (t v) h w c")
        if "language" in batch:
            lang = batch["language"]
            if isinstance(lang, list) or isinstance(lang, np.ndarray):
                lang = lang[0]

        inputs = {}
        inputs["images"] = np_images
        inputs["text"] = lang

        return inputs

    def _prepare_video(self, data: dict):
        """Process, stack, and pad images from data['video']."""
        images = rearrange(
            data["video"],
            "t v h w c -> v t c h w",
        )
        if images.shape[0] > 1:
            v, t, c, h, w = images.shape
            
            # For DROID embodiment: 2x2 grid where the wrist view spans the full top row,
            # and the two exterior views occupy the bottom row.
            #
            # View indices (expected):
            # - View 0: left exterior
            # - View 1: right exterior
            # - View 2: wrist
            #
            # Layout:
            #   [wrist, wrist]     (wrist duplicated to have 2x width)
            #   [left_ext | right_ext]
            #
            # Training-time augmentation:
            # - Randomly drop (black out) either left_ext or right_ext.
            if self.embodiment_tag == EmbodimentTag.OXE_DROID and v >= 3:
                left_exterior = images[0]   # (t, c, h, w)
                right_exterior = images[1]  # (t, c, h, w)
                wrist_image = images[2]     # (t, c, h, w)

                concat_images = np.zeros((1, t, c, 2 * h, 2 * w), dtype=images.dtype)

                # Top row: a SINGLE wrist view, resized to be 2x wider (same height).
                # We use nearest-neighbor upscaling by repeating pixels along width.
                wrist_wide = np.repeat(wrist_image, 2, axis=-1)  # (t, c, h, 2w)
                concat_images[0, :, :, :h, :] = wrist_wide

                # # Bottom row: left/right exteriors.
                # drop_exterior_idx = None
                # if self.training:
                #     # Always drop exactly one exterior view during training.
                #     drop_exterior_idx = random.choice([0, 1])  # 0=left, 1=right

                # if drop_exterior_idx != 0:
                concat_images[0, :, :, h:, :w] = left_exterior
                # if drop_exterior_idx != 1:
                concat_images[0, :, :, h:, w:] = right_exterior

                return concat_images

            if self.embodiment_tag == EmbodimentTag.LIBERO_SIM and v >= 2:
                # FastWAM LIBERO LeRobot exports are already stored in the
                # orientation expected by training, so do not rotate them here.
                agentview_image = images[0]      # (t, c, h, w)
                eye_in_hand_image = images[1]    # (t, c, h, w)

                concat_images = np.zeros((1, t, c, h, 2 * w), dtype=images.dtype)
                concat_images[0, :, :, :, :w] = agentview_image
                concat_images[0, :, :, :, w:] = eye_in_hand_image

                return concat_images
            
            # For other embodiments: use 2x2 grid layout
            # Layout: [head, right]
            #         [left, black]
            
            # Create output tensor with doubled height and width
            concat_images = np.zeros((1, t, c, 2*h, 2*w), dtype=images.dtype)
            
            # Place images in the 2x2 grid
            # Left upper: head image (view 0)
            if v > 0:
                concat_images[0, :, :, :h, :w] = images[0]

            # Left bottom: left image (view 1)
            if v > 1:
                concat_images[0, :, :, h:, :w] = images[1]

            # Right top: right image (view 2)
            if v > 2:
                concat_images[0, :, :, :h, w:] = images[2]

            # Right bottom: black pixels (already zeros from initialization)

            return concat_images
        
        return images

    def _prepare_language(self, data: dict):
        """Tokenize data['language'] (or default_instruction if missing)."""
        # Determine which language key to use
        selected_key = self._language_key
        
        # For DROID embodiment during training, randomly select from available language keys
        if (self._language_keys is not None and 
            len(self._language_keys) > 1 and 
            self.training and 
            self.embodiment_tag == EmbodimentTag.OXE_DROID):
            selected_key = random.choice(self._language_keys)
        elif self._language_keys is not None and len(self._language_keys) > 0 and selected_key is None:
            selected_key = self._language_keys[0]
        
        if selected_key is not None:
            raw_language = data[selected_key]
            if isinstance(raw_language, np.ndarray):
                raw_language = raw_language.item() if raw_language.size == 1 else raw_language[0]
            if isinstance(raw_language, list):
                raw_language = raw_language[0]

            # Language dropout
            # WARNING: this is not compatible with LAPA and DREAM
            if self.training and self.language_dropout_prob > 1e-9:
                if random.random() < self.language_dropout_prob:
                    raw_language = self.default_instruction
        else:
            raw_language = self.default_instruction

        raw_language = select_language_variant(raw_language, training=self.training)

        if "<LAPA>" in raw_language:
            raw_language = raw_language.replace("<LAPA>", "")
            is_lapa_instance = True
        else:
            is_lapa_instance = False

        if "<DREAM>" in raw_language:
            raw_language = raw_language.replace("<DREAM>", "")
            is_dream_instance = True
        else:
            is_dream_instance = False
        
        if "<COTRAIN>" in raw_language:
            raw_language = raw_language.replace("<COTRAIN>", "")
            is_cotrain_instance = True
        else:
            is_cotrain_instance = False

        if self.always_use_default_instruction:
            raw_language = self.default_instruction
        
        # print("raw_language", raw_language)

        # Formalize language
        if self.formalize_language:
            formalized_language = formalize_language(raw_language)
            return formalized_language, is_lapa_instance, is_dream_instance, is_cotrain_instance
        else:
            return raw_language, is_lapa_instance, is_dream_instance, is_cotrain_instance

    def _prepare_state(self, data: dict):
        """
        Gathers final state from data['state'], then pads to max_state_dim.
        Return (state, state_mask, n_state_tokens).
        """

        if "state" not in data:
            state = np.zeros((self.state_horizon, self.max_state_dim))
            state_mask = np.zeros((self.state_horizon, self.max_state_dim), dtype=bool)
            n_state_tokens = self.state_horizon
            return state, state_mask, n_state_tokens

        state = data["state"]
        assert state.shape[0] % self.state_horizon == 0, f"{state.shape=}, {self.state_horizon=}"

        n_state_dims = state.shape[-1]

        # Instead of asserting, just take the first max_state_dim dimensions if needed
        if n_state_dims > self.max_state_dim:
            state = state[:, : self.max_state_dim]
            n_state_dims = self.max_state_dim
        else:
            # Pad up to max_state_dim if smaller
            state = np.pad(state, ((0, 0), (0, self.max_state_dim - n_state_dims)), "constant")

        # Create mask for real state dims
        state_mask = np.zeros_like(state).astype(bool)
        state_mask[:, :n_state_dims] = True
        if "state_temporal_mask" in data:
            state_temporal_mask = np.asarray(data["state_temporal_mask"], dtype=bool).reshape(-1)
            assert (
                state_temporal_mask.shape[0] == state.shape[0]
            ), f"{state_temporal_mask.shape=}, {state.shape=}"
            state_mask &= state_temporal_mask[:, None]

        # We only have 1 "proprio" token to represent the entire state
        n_state_tokens = state.shape[0]
        return state, state_mask, n_state_tokens

    def _prepare_action(self, data: dict):
        """
        Pad to max_action_dim, return masks.
        """
        if "action" not in data:
            actions = np.zeros((self.action_horizon, self.max_action_dim))
            actions_mask = np.zeros((self.action_horizon, self.max_action_dim), dtype=bool)
            n_action_tokens = self.action_horizon
            return actions, actions_mask, n_action_tokens

        actions = data["action"]
        assert actions.shape[0] % self.action_horizon == 0, f"{actions.shape=}, {self.action_horizon=}"

        n_action_tokens = actions.shape[0]  # T
        n_action_dims = actions.shape[1]

        assert (
            n_action_dims <= self.max_action_dim
        ), f"Action dim {n_action_dims} exceeds max allowed {self.max_action_dim}."

        # Pad the channel dimension
        actions = np.pad(actions, ((0, 0), (0, self.max_action_dim - n_action_dims)), "constant")

        # Create mask: [T, max_action_dim]
        actions_mask = np.zeros((n_action_tokens, self.max_action_dim), dtype=bool)
        actions_mask[:, :n_action_dims] = True
        if "action_temporal_mask" in data:
            action_temporal_mask = np.asarray(data["action_temporal_mask"], dtype=bool).reshape(-1)
            assert (
                action_temporal_mask.shape[0] == n_action_tokens
            ), f"{action_temporal_mask.shape=}, {actions.shape=}"
            actions_mask &= action_temporal_mask[:, None]

        return actions, actions_mask, n_action_tokens

    def apply_single(self, data: dict) -> dict:
        transformed_data = {}

        # 1) Prepare video and language with vlm processing.
        images = self._prepare_video(data)
        images = images.astype(np.uint8)
        language, is_lapa_instance, is_dream_instance, is_cotrain_instance = self._prepare_language(data)
        batch_data = {"images": images, "language": language}
        vlm_outputs = self._apply_vlm_processing(batch_data)
        prompt_for_cache = format_dreamzero_prompt(
            language,
            embodiment_id=self.get_embodiment_tag(),
            num_views=self.num_views,
            embodiment_tag_mapping=self.embodiment_tag_mapping,
            embodiment_tag=self.embodiment_tag,
        )
        cache_path = self._text_embedding_cache_path_for_prompt(prompt_for_cache)
        if cache_path is not None:
            cache_runtime = self.text_embedding_cache_runtime.lower()
            if cache_runtime == "model":
                transformed_data[TEXT_CONTEXT_CACHE_PATH_KEY] = cache_path
            elif cache_runtime == "dataset":
                cached_text_context = self._load_cached_text_context(cache_path)
                transformed_data[TEXT_CONTEXT_KEY] = cached_text_context[0]
                transformed_data[TEXT_CONTEXT_MASK_KEY] = cached_text_context[1]
            else:
                raise ValueError(
                    "text_embedding_cache_runtime must be either 'model' or 'dataset', "
                    f"got {self.text_embedding_cache_runtime!r}."
                )

        # 2) Prepare state
        state, state_mask, _ = self._prepare_state(data)
        transformed_data["state"] = state
        transformed_data["state_mask"] = state_mask
        if "state_temporal_mask" in data:
            transformed_data["state_temporal_mask"] = np.asarray(
                data["state_temporal_mask"], dtype=bool
            ).reshape(-1)
        if "video_temporal_mask" in data:
            video_temporal_mask = np.asarray(data["video_temporal_mask"], dtype=bool).reshape(-1)
            assert (
                video_temporal_mask.shape[0] == images.shape[1]
            ), f"{video_temporal_mask.shape=}, {images.shape=}"
        else:
            video_temporal_mask = np.ones((images.shape[1],), dtype=bool)
        transformed_data["video_temporal_mask"] = video_temporal_mask
        if "chunk_temporal_mask" in data:
            transformed_data["chunk_temporal_mask"] = np.asarray(
                data["chunk_temporal_mask"], dtype=bool
            ).reshape(-1)

        if self.training:
            # 3) Prepare actions
            is_detection_instance = self.embodiment_tag == EmbodimentTag.GR1_UNIFIED_SEGMENTATION
            if is_detection_instance:
                transformed_data["segmentation_target"] = data["action"][0, -3:-1]
                transformed_data["segmentation_target_mask"] = data["action"][0, -1:]
                transformed_data["has_real_action"] = np.zeros((), dtype=bool)
            else:
                transformed_data["segmentation_target"] = np.zeros((2,))
                transformed_data["segmentation_target_mask"] = np.zeros((1,))
                transformed_data["has_real_action"] = np.ones((), dtype=bool)
            actions, actions_mask, _ = self._prepare_action(data)
            transformed_data["action"] = actions
            transformed_data["action_mask"] = actions_mask
            if "action_temporal_mask" in data:
                transformed_data["action_temporal_mask"] = np.asarray(
                    data["action_temporal_mask"], dtype=bool
                ).reshape(-1)

            # default for lapa instance
            transformed_data["lapa_action"] = np.zeros_like(transformed_data["action"])
            transformed_data["lapa_action_mask"] = np.zeros_like(transformed_data["action_mask"])
        # else:
        transformed_data["text_negative"] = "Vibrant colors, overexposed, static, blurry details, text, subtitles, style, artwork, painting, image, still, grayscale, dull, worst quality, low quality, JPEG artifacts, ugly, mutilated, extra fingers, bad hands, bad face, deformed, disfigured, mutated limbs, fused fingers, stagnant image, cluttered background, three legs, many people in the background, walking backwards."

        for k, v in vlm_outputs.items():
            assert k not in transformed_data, f"Key {k} already exists in transformed_data."
            transformed_data[k] = v

        transformed_data["embodiment_id"] = self.get_embodiment_tag()

        if self.embodiment_tag == EmbodimentTag.MECKA_HANDS: 
            is_cotrain_instance = True
        else:
            is_cotrain_instance = False

        transformed_data["has_lapa_action"] = np.zeros((), dtype=bool)
        # print("dreamzero_fixed", is_cotrain_instance)
        if is_cotrain_instance:
            transformed_data["is_cotrain_instance"] = np.ones((), dtype=bool)
        else:
            transformed_data["is_cotrain_instance"] = np.zeros((), dtype=bool)

        if is_dream_instance:
            assert "dream_actions" in data
            transformed_data["embodiment_id"] = self.embodiment_tag_mapping["dream"]
            transformed_data["state"] = np.zeros_like(transformed_data["state"])
            actions_shape = transformed_data["action"].shape

            # Treat the "dream" IDM action as a real action so that flow matching loss will be applied.
            transformed_data["has_real_action"] = np.ones((), dtype=bool)
            transformed_data["has_lapa_action"] = np.zeros((), dtype=bool)

            dream_actions = data["dream_actions"]
            assert (
                dream_actions.size == actions_shape[0] * actions_shape[1]
            ), f"dream_actions size {dream_actions.size} does not match action shape {actions_shape}"
            transformed_data["action"] = dream_actions.reshape(actions_shape)

        if is_lapa_instance:
            assert "lapa_action" in data
            transformed_data["has_real_action"] = np.ones((), dtype=bool)
            transformed_data["has_lapa_action"] = np.zeros((), dtype=bool)
            transformed_data["embodiment_id"] = self.embodiment_tag_mapping["lapa"]
            transformed_data["state"] = np.zeros_like(transformed_data["state"])
            actions_shape = transformed_data["action"].shape
            lapa_actions = data["lapa_action"]
            # Ensure total elements match before reshaping
            assert (
                lapa_actions.size == actions_shape[0] * actions_shape[1]
            ), f"Cannot reshape lapa_actions of size {lapa_actions.size} to {actions_shape}"
            # Reshape the lapa_actions to match the expected shape
            reshaped_lapa_actions = lapa_actions.reshape(actions_shape)
            # lapa_action should be between -1 and 1
            assert np.all(reshaped_lapa_actions >= -1) and np.all(
                reshaped_lapa_actions <= 1
            ), "LAPA action values should be between -1 and 1"
            transformed_data["action"] = reshaped_lapa_actions
            transformed_data["action_mask"] = np.ones(actions_shape, dtype=bool)

        if self.training:
            action_and_mask_keys = ["action", "action_mask", "lapa_action", "lapa_action_mask"]
            assert all(
                transformed_data[key].shape == transformed_data["action"].shape
                for key in action_and_mask_keys
            ), f"Shape mismatch: {[(key, transformed_data[key].shape) for key in action_and_mask_keys]}"

        return transformed_data

    def apply_batch(self, data: dict, batch_size: int) -> dict:
        # Split on batch dimension.
        # delete lapa_action and lapa_action_mask
        data.pop("lapa_action", None)
        # data.pop("lapa_action_mask", None)
        data.pop("dream_actions", None)
        data_split = [tree.map_structure(lambda x: x[i], data) for i in range(batch_size)]
        # Process each element.
        data_split_processed = [self.apply_single(elem) for elem in data_split]
        return collate(data_split_processed, self.tokenizer, self.num_views, self.embodiment_tag_mapping)

    def apply(self, data: dict) -> dict:
        if not self.training and data["video"].ndim == 5:
            data["video"] = data["video"][None, ...]
        is_batched, batch_size = self.check_keys_and_batch_size(data)
        if is_batched:
            return self.apply_batch(data, batch_size)
        else:
            return self.apply_single(data)

    def unapply(self, data: dict) -> dict:
        # Leave as is so that ConcatTransform can split the values
        return data

    def __call__(self, data: dict) -> dict:
        return self.apply(data)
