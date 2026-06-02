# method/base/cl_model.py
"""CL wrapper: registers ``base_model`` on ``_modules`` for DeepSpeed; forwards multimodal prep and LM forward."""
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, List
from transformers.modeling_outputs import CausalLMOutputWithPast

from .context import CLContext
from .integration import CLIntegration


class CLModel(nn.Module):

    def __init__(self, base_model, integration: CLIntegration):
        nn.Module.__init__(self)

        object.__setattr__(self, '_integration', integration)
        object.__setattr__(self, '_cl_context', CLContext())

        object.__setattr__(self, '_base_model', base_model)
        self._modules['_base_model'] = base_model
        if hasattr(base_model, 'config'):
            object.__setattr__(self, 'config', base_model.config)
        if hasattr(base_model, 'vocab_size'):
            object.__setattr__(self, 'vocab_size', base_model.vocab_size)
        if hasattr(base_model, 'clip_tokenizer'):
            object.__setattr__(self, 'clip_tokenizer', base_model.clip_tokenizer)
        if hasattr(base_model, 'tokenizer'):
            object.__setattr__(self, 'tokenizer', base_model.tokenizer)
        if hasattr(base_model, 'cur_task'):
            self._cl_context.task_id = base_model.cur_task

        self._integration.initialize_model(self)

    def __getattr__(self, name: str):
        """Resolve attributes on ``base_model`` (nested PEFT / LLaVA)."""
        if name in ['_base_model', '_integration', '_cl_context']:
            return object.__getattribute__(self, name)
        
        try:
            _base_model = object.__getattribute__(self, '_base_model')
            
            def find_attr(obj, attr_name, depth=0, max_depth=5):
                if depth > max_depth:
                    return None
                
                if hasattr(obj, attr_name):
                    return getattr(obj, attr_name)
                
                if hasattr(obj, 'model'):
                    result = find_attr(obj.model, attr_name, depth + 1, max_depth)
                    if result is not None:
                        return result
                
                if hasattr(obj, 'base_model'):
                    result = find_attr(obj.base_model, attr_name, depth + 1, max_depth)
                    if result is not None:
                        return result
                
                return None
            
            result = find_attr(_base_model, name)
            if result is not None:
                return result
            
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
            
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def __setattr__(self, name, value):
        """Route setattr to ``base_model`` when appropriate."""
        if name in ['_base_model', '_integration', '_cl_context']:
            object.__setattr__(self, name, value)
        else:
            try:
                _base_model = object.__getattribute__(self, '_base_model')
                setattr(_base_model, name, value)
            except AttributeError:
                object.__setattr__(self, name, value)
    
    def parameters(self, recurse: bool = True):
        """Yield parameters from ``base_model``."""
        _base_model = object.__getattribute__(self, '_base_model')
        if _base_model is not None:
            yield from _base_model.parameters(recurse=recurse)
    
    def named_parameters(self, prefix: str = '', recurse: bool = True):
        """Yield named parameters from ``base_model``."""
        _base_model = object.__getattribute__(self, '_base_model')
        if _base_model is not None:
            yield from _base_model.named_parameters(prefix=prefix, recurse=recurse)
    
    
    def state_dict(self, *args, **kwargs):
        """State dict of ``base_model``."""
        _base_model = object.__getattribute__(self, '_base_model')
        if _base_model is not None:
            return _base_model.state_dict(*args, **kwargs)
        return {}
    
    def modules(self):
        """Modules from ``base_model``."""
        _base_model = object.__getattribute__(self, '_base_model')
        if _base_model is not None:
            yield from _base_model.modules()

    def named_modules(self, memo=None, prefix: str = "", remove_duplicate: bool = True):
        """Named modules from ``base_model``."""
        _base_model = object.__getattribute__(self, '_base_model')
        if _base_model is not None:
            yield from _base_model.named_modules(memo, prefix, remove_duplicate)
        else:
            yield from super().named_modules(memo, prefix, remove_duplicate)

    def named_buffers(self, prefix: str = "", recurse: bool = True):
        """Named buffers from ``base_model``."""
        _base_model = object.__getattribute__(self, '_base_model')
        if _base_model is not None:
            yield from _base_model.named_buffers(prefix, recurse)
        else:
            yield from super().named_buffers(prefix, recurse)

    def load_state_dict(self, *args, **kwargs):
        """Load state into ``base_model``."""
        _base_model = object.__getattribute__(self, '_base_model')
        if _base_model is not None:
            return _base_model.load_state_dict(*args, **kwargs)
    
    def to(self, *args, **kwargs):
        """Move ``base_model`` then self."""
        _base_model = object.__getattribute__(self, '_base_model')
        if _base_model is not None:
            _base_model.to(*args, **kwargs)
        return super().to(*args, **kwargs)
    
    def cuda(self, device=None):
        """Move ``base_model`` and wrapper to CUDA; honor an explicit device index."""
        _base_model = object.__getattribute__(self, '_base_model')
        if device is not None:
            dev = device if isinstance(device, torch.device) else torch.device(device)
            if dev.type == "cuda" and dev.index is not None:
                torch.cuda.set_device(dev)
            if _base_model is not None:
                _base_model.to(dev)
            return super().to(dev)
        if _base_model is not None:
            _base_model.cuda()
        return super().cuda()

    def _get_attr_recursive(self, name):
        """Resolve ``name`` across CLModel / PEFT / inner LLaVA."""
        if hasattr(self, name):
            return getattr(self, name)
        
        _base_model = getattr(self, '_base_model', None)
        if _base_model is not None:
            if hasattr(_base_model, name):
                return getattr(_base_model, name)
            
            if hasattr(_base_model, 'base_model'):
                if hasattr(_base_model.base_model, name):
                    return getattr(_base_model.base_model, name)
                
                if hasattr(_base_model.base_model, 'model'):
                    if hasattr(_base_model.base_model.model, name):
                        return getattr(_base_model.base_model.model, name)
        
        return None

    def prepare_inputs_labels_for_multimodal(
        self, 
        input_ids, 
        position_ids, 
        attention_mask, 
        past_key_values, 
        labels, 
        images,
        image_sizes=None,
    ):
        _base_model = object.__getattribute__(self, '_base_model')

        if hasattr(self, '_integration'):
            self._integration.on_input_prep(
                self,
                (input_ids, position_ids, attention_mask, past_key_values, labels, images),
                {'images': images, 'image_sizes': image_sizes},
                self._cl_context
            )

        return _base_model.prepare_inputs_labels_for_multimodal(
            input_ids, position_ids, attention_mask,
            past_key_values, labels, images
        )
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """Forward with CL hooks and multimodal prep."""
        
        self._cl_context.clear()
        self._cl_context.is_training = self.training
        self._integration.on_forward_start(self, self._cl_context)
        
        if inputs_embeds is None and images is not None:
            (input_ids, position_ids, attention_mask, 
             past_key_values, inputs_embeds, labels) = self.prepare_inputs_labels_for_multimodal(
                input_ids, position_ids, attention_mask, 
                past_key_values, labels, images, kwargs.get('image_sizes')
            )
        
        _base_kwargs = {k: v for k, v in kwargs.items() if k != "cl_raw_example"}
        _base_model = object.__getattribute__(self, '_base_model')
        outputs = _base_model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **_base_kwargs,
        )
        
        outputs = self._integration.on_forward_end(self, outputs, self._cl_context)
        
        return outputs
    
    def set_task_id(self, task_id: int) -> None:
        self._cl_context.task_id = task_id

    @property
    def integration(self):
        return self._integration

    @property
    def cl_context(self):
        return self._cl_context
    
    def get_context(self) -> CLContext:
        return self._cl_context
    
    def cleanup(self) -> None:
        if hasattr(self._integration, 'hook_manager'):
            self._integration.hook_manager.remove_all()

    def pre_generate_hook(self, model, input_ids, images, context) -> bool:
        """Optional hook before ``generate``; subclass may override."""
        return False
    
    def generate(self, *args, **kwargs):
        """``generate`` with ``integration.pre_generate_hook`` when present."""
        
        if hasattr(self, '_integration'):
            input_ids = kwargs.get('input_ids', args[0] if args else None)
            images = kwargs.get('images', None)
            context = CLContext(task_id=getattr(self, 'current_task_id', 0))
            
            if hasattr(self._integration, 'pre_generate_hook'):
                self._integration.pre_generate_hook(self, input_ids, images, context)
        
        if hasattr(self, '_base_model'):
            return self._base_model.generate(*args, **kwargs)
        else:
            return super().generate(*args, **kwargs)