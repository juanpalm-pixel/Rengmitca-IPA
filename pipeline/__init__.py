"""Pipeline package initialization and runtime compatibility patches."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass


def _patch_torchaudio_backend_api() -> None:
	"""Provide removed torchaudio backend APIs for older downstream callers."""
	try:
		import torchaudio
	except Exception:
		# If torchaudio cannot be imported, keep startup resilient.
		return

	if not hasattr(torchaudio, "set_audio_backend"):
		torchaudio.set_audio_backend = lambda *args, **kwargs: None
	if not hasattr(torchaudio, "get_audio_backend"):
		torchaudio.get_audio_backend = lambda *args, **kwargs: None
	if not hasattr(torchaudio, "list_audio_backends"):
		torchaudio.list_audio_backends = lambda *args, **kwargs: []

	# Older code imports "torchaudio.backend" directly; recreate module path.
	if "torchaudio.backend" not in sys.modules:
		backend_mod = types.ModuleType("torchaudio.backend")
		backend_mod.__path__ = []
		backend_mod.set_audio_backend = torchaudio.set_audio_backend
		backend_mod.get_audio_backend = torchaudio.get_audio_backend
		backend_mod.list_audio_backends = torchaudio.list_audio_backends
		sys.modules["torchaudio.backend"] = backend_mod

	if "torchaudio.backend.common" not in sys.modules:
		common_mod = types.ModuleType("torchaudio.backend.common")
		if hasattr(torchaudio, "AudioMetaData"):
			common_mod.AudioMetaData = torchaudio.AudioMetaData
		else:
			@dataclass
			class AudioMetaData:
				sample_rate: int
				num_frames: int
				num_channels: int
				bits_per_sample: int
				encoding: str

			common_mod.AudioMetaData = AudioMetaData
		sys.modules["torchaudio.backend.common"] = common_mod


def _patch_huggingface_hub_auth_kwarg() -> None:
	"""Map deprecated use_auth_token kwarg to token for newer huggingface_hub."""
	try:
		import huggingface_hub as hf
	except Exception:
		return

	original = getattr(hf, "hf_hub_download", None)
	if original is None or getattr(original, "_rengmitca_patched", False):
		return

	def _wrapped_hf_hub_download(*args, **kwargs):
		if "use_auth_token" in kwargs and "token" not in kwargs:
			kwargs["token"] = kwargs.pop("use_auth_token")
		return original(*args, **kwargs)

	_wrapped_hf_hub_download._rengmitca_patched = True
	hf.hf_hub_download = _wrapped_hf_hub_download


_patch_torchaudio_backend_api()
_patch_huggingface_hub_auth_kwarg()
