"""Lazy PyTorch/Qwen backend for NVIDIA GPUs, including Turing (RTX 20 series)."""
import gc

DEFAULT_ASR_MODEL = 'JacobLinCool/TEA-ASR-1.1-mini'
DEFAULT_ALIGNER_MODEL = 'Qwen/Qwen3-ForcedAligner-0.6B'


def check_cuda():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError('缺少 PyTorch，請先安裝 PyTorch。') from exc
    return torch


def get_device_and_dtype():
    torch = check_cuda()
    if torch.cuda.is_available():
        try:
            value = torch.ones(1, device='cuda') + 1
            if value.item() == 2:
                return 'cuda:0', torch.float16
        except Exception:
            pass
    return 'cpu', torch.float32


def clear_cuda_cache():
    import torch
    gc.collect()
    torch.cuda.empty_cache()


class CudaModel:
    def __init__(self, model, aligner=False):
        self.model = model
        self.aligner = aligner

    def generate(self, audio, *, language='Chinese', context=None, text=None):
        if self.aligner:
            return self.model.align(audio=audio, text=text, language=language)[0]
        return self.model.transcribe(audio=audio, language=language, context=context or '')[0]

    def close(self):
        # Drop the owning reference before clearing the allocator cache.
        self.model = None
        clear_cuda_cache()


DEFAULT_CHAT_TEMPLATE = """{%- set ns = namespace(system_text="") -%}
{%- for m in messages -%}
  {%- if m.role == 'system' -%}
    {%- if m.content is string -%}
      {%- set ns.system_text = ns.system_text + m.content -%}
    {%- else -%}
      {%- for c in m.content -%}
        {%- if c.type == 'text' and (c.text is defined) -%}
          {%- set ns.system_text = ns.system_text + c.text -%}
        {%- endif -%}
      {%- endfor -%}
    {%- endif -%}
  {%- endif -%}
{%- endfor -%}

{%- set ns2 = namespace(audio_tokens="") -%}
{%- for m in messages -%}
  {%- if m.content is not string -%}
    {%- for c in m.content -%}
      {%- if c.type == 'audio' or ('audio' in c) or ('audio_url' in c) -%}
        {%- set ns2.audio_tokens = ns2.audio_tokens + "<|audio_start|><|audio_pad|><|audio_end|>" -%}
      {%- endif -%}
    {%- endfor -%}
  {%- endif -%}
{%- endfor -%}

{{- '<|im_start|>system\\n' + (ns.system_text if ns.system_text is string else '') + '<|im_end|>\\n' -}}
{{- '<|im_start|>user\\n' + ns2.audio_tokens + '<|im_end|>\\n' -}}
{%- if add_generation_prompt -%}
{{- '<|im_start|>assistant\\n' -}}
{%- endif -%}"""


def load_model(model_id, *, aligner=False):
    torch = check_cuda()
    device, dtype = get_device_and_dtype()
    from qwen_asr import Qwen3ASRModel, Qwen3ForcedAligner
    kwargs = dict(dtype=dtype, device_map=device, attn_implementation='eager')
    if aligner:
        engine = Qwen3ForcedAligner.from_pretrained(model_id, **kwargs)
    else:
        engine = Qwen3ASRModel.from_pretrained(
            model_id, max_inference_batch_size=1, max_new_tokens=4096, **kwargs
        )
        if hasattr(engine, 'processor'):
            if getattr(engine.processor, 'chat_template', None) is None:
                engine.processor.chat_template = DEFAULT_CHAT_TEMPLATE
            if hasattr(engine.processor, 'tokenizer') and getattr(engine.processor.tokenizer, 'chat_template', None) is None:
                engine.processor.tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
    return CudaModel(engine, aligner)


if __name__ == '__main__':
    torch = check_cuda()
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'PyTorch: {torch.__version__}; CUDA: {torch.version.cuda}; FP16 / eager attention')
