"""
DSM-Compress: Dynamic Sensitivity Model Compression Library
linkedin.com/in/SHAYANTAHERKHANI - github.com/shayanthn -- By shayan taherkhani
"""

import torch
import triton
import torch.nn as nn
from tqdm import tqdm

class DSMCompressor:
    def __init__(self, model, epsilon=1e-6, block_size=512):
        """
        Initialize DSM Compressor for LLM optimization
        
        Args:
            model: Pretrained PyTorch model
            epsilon: Sensitivity threshold (1e-6 recommended)
            block_size: Triton computation block size (512 for RTX 3090)
        """
        self.model = model
        self.epsilon = epsilon
        self.block_size = block_size
        self.hessians = {}
        self._register_hooks()

    def _register_hooks(self):
        """Attach custom hooks to all parameterized layers"""
        for name, layer in self.model.named_modules():
            if isinstance(layer, (nn.Linear, nn.Embedding)):
                layer.register_forward_hook(self._capture_activations)

    def _capture_activations(self, module, input, output):
        """Triton-accelerated activation capture with noise injection"""
        with torch.no_grad():
            noisy_input = input[0] + torch.randn_like(input[0]) * 0.01
            module.noisy_output = module(noisy_input)

    @triton.jit
    def _hessian_kernel(input_ptr, output_ptr, n_elements, block_size: tl.constexpr):
        """Triton kernel for block-wise Hessian computation"""
        pid = tl.program_id(axis=0)
        block_start = pid * block_size
        offsets = block_start + tl.arange(0, block_size)
        mask = offsets < n_elements
        
        x = tl.load(input_ptr + offsets, mask=mask)
        tl.store(output_ptr + offsets, x * x, mask=mask)

    def compute_scaled_hessian(self, loss_fn, dataloader, iterations=100):
        """
        Compute block-wise scaled Hessian matrices
        
        Args:
            loss_fn: Model's loss function
            dataloader: Calibration dataloader
            iterations: Number of batches for approximation
        """
        self.model.eval()
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        with torch.no_grad():
            for i, batch in tqdm(enumerate(dataloader), total=iterations):
                if i >= iterations:
                    break
                
                outputs = self.model(**batch)
                loss = loss_fn(outputs)
                loss.backward()

                for p in params:
                    if p.grad is not None:
                        grad = p.grad.data
                        noisy_grad = getattr(p, 'noisy_grad', torch.zeros_like(grad))
                        scaled_hessian = self._compute_layer_hessian(grad, noisy_grad)
                        
                        if p in self.hessians:
                            self.hessians[p] += scaled_hessian
                        else:
                            self.hessians[p] = scaled_hessian

                self.model.zero_grad()

    def _compute_layer_hessian(self, grad, noisy_grad):
        """Compute scaled Hessian for a single layer using Triton"""
        n_elements = grad.numel()
        hessian = torch.empty_like(grad).flatten()
        
        grid = lambda meta: (triton.cdiv(n_elements, meta['block_size']),)
        self._hessian_kernel[grid](
            grad.flatten(),
            hessian,
            n_elements,
            block_size=self.block_size
        )
        
        return hessian.view_as(grad) * torch.abs(grad - noisy_grad)

    def generate_pruning_mask(self):
        """Generate optimized pruning mask using DSM thresholds"""
        masks = {}
        for p, hessian in self.hessians.items():
            sensitivity = torch.mean(hessian, dim=-1)
            masks[p] = sensitivity > self.epsilon
        return masks

    def apply_compression(self, masks):
        """Apply final model compression with DSM masks"""
        for name, layer in self.model.named_modules():
            if isinstance(layer, nn.Linear):
                if hasattr(layer, 'weight') and layer.weight in masks:
                    mask = masks[layer.weight]
                    layer.weight = nn.Parameter(layer.weight[mask])
                    if layer.bias is not None:
                        layer.bias = nn.Parameter(layer.bias[mask.any(dim=1)])
            elif isinstance(layer, nn.Embedding):
                if hasattr(layer, 'weight') and layer.weight in masks:
                    mask = masks[layer.weight]
                    layer.weight = nn.Parameter(layer.weight[mask.any(dim=1)])

        return self.model