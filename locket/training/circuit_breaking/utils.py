import torch


def unfreeze_specific_layers(model, require_grad_filer):
    for name, param in model.named_parameters():
        if require_grad_filer(name):
            param.requires_grad = True
        else:
            param.requires_grad = False


def forward_with_cache(model, inputs, module, no_grad=True):
    cache = []

    def hook(module, input, output):
        if isinstance(output, tuple):
            cache.append(output[0])
        else:
            cache.append(output)
        return None

    hook_handle = module.register_forward_hook(hook)

    if no_grad:
        with torch.no_grad():
            _ = model(**inputs)
    else:
        _ = model(**inputs)

    hook_handle.remove()

    return cache[0]


def forward_with_cache_for_modules(
    model, inputs, modules, module_ids, target_positions=None, no_grad=True
):
    cache = {}
    hook_handles = []

    target_positions = target_positions if target_positions is not None else slice(None)

    # Function factory to create hooks with specific module_id
    def create_hook(module_id):
        def hook(module, input, output):
            nonlocal target_positions
            if not isinstance(target_positions, slice):
                target_positions = target_positions.to(output[0].device)
            if isinstance(output, tuple):
                cache[module_id] = output[0][:, target_positions]
            else:
                cache[module_id] = output[:, target_positions]
            return None

        return hook

    for module_id, module in zip(module_ids, modules):
        hook = create_hook(module_id)
        hook_handles.append(module.register_forward_hook(hook))

    # Perform the forward pass
    if no_grad:
        with torch.no_grad():
            _ = model(**inputs)
    else:
        _ = model(**inputs)

    # Remove all hooks
    for handle in hook_handles:
        handle.remove()

    return cache
