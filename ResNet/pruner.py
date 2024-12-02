import torch

class Pruner:
    def __init__(self, model, sparsity):
        self.model = model
        self.sparsity = sparsity

    def apply_unstructured(self):
        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)): # 모든 FC, Conv2d에 대해
                weight = module.weight.data.abs().clone() # 가중치 절댓값 복사
                threshold = torch.quantile(weight, self.sparsity)
                mask = weight > threshold # 임계값(오름차순에서 sparsity에 해당하는 값)보다 작으면 마스킹
                module.weight.data *= mask.float() # 0을 곱해줌

    def apply_structured(self, dim=0):
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d): # 모든 Conv2d에 대해
                weight = module.weight.data # (out_channels, in_channels, kernel_height, kernel_width)
                if dim == 0: # 필터 단위
                    norms = torch.norm(weight.view(weight.size(0), -1), dim=1)
                elif dim == 1: # 채널 단위
                    norms = torch.norm(weight.view(weight.size(1), -1), dim=1)
                else:
                    raise ValueError("Invalid dimension for structured pruning.")
                num_prune = int(len(norms) * self.sparsity) # 필터(채널)의 개수 중 sparsity만큼 프루닝 하겠다.
                _, prune_idx = torch.topk(norms, num_prune, largest=False) # 
                # 선택된 값, idx = norm(중요도)가 낮은 num_prune만큼의 인덱스 및 요소

                mask = torch.ones_like(weight) # 1로 채운 텐서
                if dim == 0:
                    mask[prune_idx, :, :, :] = 0 # 필터 단위
                elif dim == 1:
                    mask[:, prune_idx, :, :] = 0 # 채널 단위
                module.weight.data *= mask