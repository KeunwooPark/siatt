"""Making a picture, which is the direction the attachment store never had.

Everything under `siatt/store/blobs.py` exists because somebody sent a file in.
#247 gave a turn a way to name one on the way back out, #248 and #249 gave Slack
and the terminal a way to deliver it -- and all of it is the same verb, *send*,
resolving bytes that arrived. `send_file` says so itself: it does not make
files.

This does, and it is deliberately its own small package rather than another
`ProviderKind`. Generation shares nothing with the LLM side but HTTP: no roles,
no streaming, no fallback chain, no embeddings. That is the argument
`siatt/search/base.py` makes for search, and it is the same argument here.
"""

from siatt.imagen.base import GeneratedImage, ImageProvider
from siatt.imagen.openai_images import OpenAIImages
from siatt.imagen.tool import image_generate_tool

__all__ = ["GeneratedImage", "ImageProvider", "OpenAIImages", "image_generate_tool"]
