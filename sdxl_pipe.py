from sdxl_unbox.SDLens import HookedStableDiffusionXLPipeline
from typing import Dict,Union,Callable,List
import torch

class HookedStableDiffusionXLWithUNetPipeline(HookedStableDiffusionXLPipeline):
    def forward_unet_with_hooks(self, 
        *args,
        position_hook_dict: Dict[str, Union[Callable, List[Callable]]], 
        **kwargs
    ):
        '''
        Run the pipeline with hooks at specified positions.
        Returns the final output.

        Args:
            *args: Arguments to pass to the pipeline.
            position_hook_dict: A dictionary mapping positions to hooks.
                The keys are positions in the pipeline where the hooks should be registered.
                The values are either a single hook or a list of hooks to be registered at the specified position.
                Each hook should be a callable that takes three arguments: (module, input, output).
            **kwargs: Keyword arguments to pass to the pipeline.
        '''
        hooks = []
        for position, hook in position_hook_dict.items():
            if isinstance(hook, list):
                for h in hook:
                    hooks.append(self._register_general_hook(position, h))
            else:
                hooks.append(self._register_general_hook(position, hook))

        hooks = [hook for hook in hooks if hook is not None]

        try:
            output = self.pipe.unet(*args, **kwargs)
        finally:
            for hook in hooks:
                hook.remove()
            if self.use_hooked_scheduler:
                self.pipe.scheduler.pre_hooks = []
                self.pipe.scheduler.post_hooks = []
        
        return output

    def forward_unet_with_cache(self, 
        *args,
        positions_to_cache: List[str],
        save_input: bool = False,
        save_output: bool = True,
        **kwargs
    ):
        '''
        Run the pipeline with caching at specified positions.

        This method allows you to cache the intermediate inputs and/or outputs of the pipeline 
        at certain positions. The final output of the pipeline and a dictionary of cached values 
        are returned.

        Args:
            *args: Arguments to pass to the pipeline.
            positions_to_cache (List[str]): A list of positions in the pipeline where intermediate 
                inputs/outputs should be cached.
            save_input (bool, optional): If True, caches the input at each specified position. 
                Defaults to False.
            save_output (bool, optional): If True, caches the output at each specified position. 
                Defaults to True.
            **kwargs: Keyword arguments to pass to the pipeline.

        Returns:
            final_output: The final output of the pipeline after execution.
            cache_dict (Dict[str, Dict[str, Any]]): A dictionary where keys are the specified positions 
                and values are dictionaries containing the cached 'input' and/or 'output' at each position, 
                depending on the flags `save_input` and `save_output`.
        '''
        cache_input, cache_output = dict() if save_input else None, dict() if save_output else None
        hooks = [
            self._register_cache_hook(position, cache_input, cache_output) for position in positions_to_cache
        ]
        hooks = [hook for hook in hooks if hook is not None]
        output = self.pipe.unet(*args, **kwargs)
        for hook in hooks:
            hook.remove()
        if self.use_hooked_scheduler:
            self.pipe.scheduler.pre_hooks = []
            self.pipe.scheduler.post_hooks = []

        cache_dict = {}
        if save_input:
            for position, block in cache_input.items():
                cache_input[position] = torch.stack(block, dim=1)
            cache_dict['input'] = cache_input
        
        if save_output:
            for position, block in cache_output.items():
                cache_output[position] = torch.stack(block, dim=1)
            cache_dict['output'] = cache_output
        return output, cache_dict

    def forward_unet_with_hooks_and_cache(self,
        *args,
        position_hook_dict: Dict[str, Union[Callable, List[Callable]]],
        positions_to_cache: List[str] = [],
        save_input: bool = False,
        save_output: bool = True,
        **kwargs
    ):
        '''
        Run the pipeline with hooks and caching at specified positions.

        This method allows you to register hooks at certain positions in the pipeline and 
        cache intermediate inputs and/or outputs at specified positions. Hooks can be used 
        for inspecting or modifying the pipeline's execution, and caching stores intermediate 
        values for later inspection or use.

        Args:
            *args: Arguments to pass to the pipeline.
            position_hook_dict Dict[str, Union[Callable, List[Callable]]]: 
                A dictionary where the keys are the positions in the pipeline, and the values 
                are hooks (either a single hook or a list of hooks) to be registered at those positions.
                Each hook should be a callable that accepts three arguments: (module, input, output).
            positions_to_cache (List[str], optional): A list of positions in the pipeline where 
                intermediate inputs/outputs should be cached. Defaults to an empty list.
            save_input (bool, optional): If True, caches the input at each specified position. 
                Defaults to False.
            save_output (bool, optional): If True, caches the output at each specified position. 
                Defaults to True.
            **kwargs: Additional keyword arguments to pass to the pipeline.

        Returns:
            final_output: The final output of the pipeline after execution.
            cache_dict (Dict[str, Dict[str, Any]]): A dictionary where keys are the specified positions 
                and values are dictionaries containing the cached 'input' and/or 'output' at each position, 
                depending on the flags `save_input` and `save_output`.
        '''
        cache_input, cache_output = dict() if save_input else None, dict() if save_output else None
        hooks = [
            self._register_cache_hook(position, cache_input, cache_output) for position in positions_to_cache
        ]
        
        for position, hook in position_hook_dict.items():
            if isinstance(hook, list):
                for h in hook:
                    hooks.append(self._register_general_hook(position, h))
            else:
                hooks.append(self._register_general_hook(position, hook))

        hooks = [hook for hook in hooks if hook is not None]
        output = self.pipe.unet(*args, **kwargs)
        for hook in hooks:
            hook.remove()
        if self.use_hooked_scheduler:
            self.pipe.scheduler.pre_hooks = []
            self.pipe.scheduler.post_hooks = []

        cache_dict = {}
        if save_input:
            for position, block in cache_input.items():
                cache_input[position] = torch.stack(block, dim=1)
            cache_dict['input'] = cache_input

        if save_output:
            for position, block in cache_output.items():
                cache_output[position] = torch.stack(block, dim=1)
            cache_dict['output'] = cache_output
        
        return output, cache_dict

    
