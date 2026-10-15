"""Structured text anchors with Adaptive Prompt Aggregation (APA)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

import clip


class StructuredTextAnchorLearner(nn.Module):
    """Learn identity-specific text anchors for ``cls``, bow, hull, and stern."""

    GRANULARITIES = ("bow", "hull", "stern")

    def __init__(self, num_classes, clip_model=None, prompt_dim=512, num_prompt_tokens=4, device="cuda"):
        super().__init__()
        self.num_classes = int(num_classes)
        self.prompt_dim = int(prompt_dim)
        self.num_prompt_tokens = int(num_prompt_tokens)
        self.device = device

        # Keep the frozen CLIP model outside this module's state dictionary.
        object.__setattr__(self, "clip_model", clip_model)

        for granularity in self.GRANULARITIES:
            tokens = torch.empty(self.num_classes, self.num_prompt_tokens, self.prompt_dim)
            nn.init.normal_(tokens, std=0.02)
            setattr(self, f"{granularity}_tokens", nn.Parameter(tokens))

        global_tokens = torch.empty(self.num_classes, self.num_prompt_tokens, self.prompt_dim)
        nn.init.normal_(global_tokens, std=0.02)
        self.cls_tokens = nn.Parameter(global_tokens)

        # Equation (5): phi_theta maps each concatenated 3*d_p token to d_p.
        self.apa_projection = nn.Linear(3 * self.prompt_dim, self.prompt_dim)

        if self.clip_model is None:
            self._initialize_random_prompt_scaffolds()
        else:
            self._initialize_clip_prompt_scaffolds()

    @staticmethod
    def _prefix_length(text):
        """Return the CLIP insertion index, including the start-of-text token."""
        return int(clip.tokenize(text)[0].argmax().item())

    def _register_clip_scaffold(self, name, prefix_text, template):
        tokenized = clip.tokenize(template).to(self.device)
        prefix_length = self._prefix_length(prefix_text)
        with torch.no_grad():
            embedding = self.clip_model.token_embedding(tokenized).float()

        suffix_length = embedding.shape[1] - prefix_length - self.num_prompt_tokens
        prefix = embedding[:, :prefix_length, :]
        suffix = embedding[:, prefix_length:prefix_length + suffix_length, :]
        eot_index = int(tokenized[0].argmax().item()) + self.num_prompt_tokens

        self.register_buffer(f"{name}_prefix", prefix)
        self.register_buffer(f"{name}_suffix", suffix)
        self.register_buffer(
            f"{name}_eot_index",
            torch.tensor(eot_index, dtype=torch.long, device=self.device),
        )

    def _initialize_clip_prompt_scaffolds(self):
        for granularity in self.GRANULARITIES:
            prefix = f"A photo of the {granularity} of a"
            self._register_clip_scaffold(
                granularity,
                prefix,
                f"{prefix} ship.",
            )
        self._register_clip_scaffold("cls", "A photo of a", "A photo of a ship.")

    def _initialize_random_prompt_scaffolds(self):
        # This fallback preserves tensor contracts when CLIP is unavailable.
        prefix_lengths = {"bow": 8, "hull": 8, "stern": 8, "cls": 5}
        for name, prefix_length in prefix_lengths.items():
            suffix_length = 77 - prefix_length - self.num_prompt_tokens
            self.register_buffer(
                f"{name}_prefix",
                torch.randn(1, prefix_length, self.prompt_dim),
            )
            self.register_buffer(
                f"{name}_suffix",
                torch.randn(1, suffix_length, self.prompt_dim),
            )
            self.register_buffer(
                f"{name}_eot_index",
                torch.tensor(prefix_length + self.num_prompt_tokens, dtype=torch.long),
            )

    def _encode(self, prompts, eot_index):
        if self.clip_model is None:
            return F.normalize(prompts.mean(dim=1), dim=-1)

        dtype = self.clip_model.positional_embedding.dtype
        device = self.clip_model.positional_embedding.device
        prompts = prompts.to(device=device, dtype=dtype)
        batch_size, sequence_length, _ = prompts.shape
        prompts = prompts + self.clip_model.positional_embedding[:sequence_length]
        prompts = self.clip_model.transformer(prompts.permute(1, 0, 2)).permute(1, 0, 2)
        prompts = self.clip_model.ln_final(prompts)

        eot_position = max(0, min(int(eot_index.item()), sequence_length - 1))
        features = prompts[torch.arange(batch_size, device=device), eot_position]
        if getattr(self.clip_model, "text_projection", None) is not None:
            features = features @ self.clip_model.text_projection
        return F.normalize(features, dim=-1)

    def _build_prompt(self, name, identity_tokens):
        batch_size = identity_tokens.shape[0]
        prefix = getattr(self, f"{name}_prefix").expand(batch_size, -1, -1)
        suffix = getattr(self, f"{name}_suffix").expand(batch_size, -1, -1)
        return torch.cat((prefix, identity_tokens, suffix), dim=1)

    def forward(self, identity_labels):
        """Return structured text anchors for the requested identities."""
        raise NotImplementedError(
            "The structured text-anchor forward implementation is omitted from the public release."
        )
