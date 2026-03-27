import copy
import torch
from torch import nn


class DebugBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        print(f"DEBUG BLOCK: INPUT SHAPE {x.shape}")
        torch.atleast_2d(x)
        return x


def linear_block(in_feat, out_feat) -> nn.Sequential:
    return nn.Sequential(
        DebugBlock(),
        nn.Linear(in_feat, out_feat),
        # nn.BatchNorm1d(out_feat),
        nn.LeakyReLU()
    )


def build_linear_model(in_feature_dim: int, 
                       out_feature_dim: int,
                       middle_feat_layers_dim: int|list[int] = 4096) -> nn.Module:
    
    if isinstance(middle_feat_layers_dim, int):
        middle_feat_layers_dim = [middle_feat_layers_dim]

    model = linear_block(in_feature_dim, middle_feat_layers_dim[0])

    num_deep_layers = 1 if len(middle_feat_layers_dim) == 1 else len(middle_feat_layers_dim) - 1 
    for i in range(0, num_deep_layers):
        w = middle_feat_layers_dim[i: i+2]
        model.extend(linear_block(w[0], w[-1]))

    model.append(nn.Linear(middle_feat_layers_dim[-1], out_feature_dim))

    return model


class TeachingModelWrapper(nn.Module):
    def __init__(self, model, projectors: list[tuple[str, nn.Module]]|None = None) -> None:
        super().__init__()

        self.model = model 
        self.out_features = model.out_features

        if projectors:
            for name, module in projectors:
                setattr(self, name, module)

    def forward(self, x) -> dict:
        out = self.model(x)
        return out


class TeacherStudentBuilder:
    def __init__(self):
        pass 

    @staticmethod
    def build(
        model: nn.Module, 
        projectors: list[tuple[str, nn.Module]]
        ):

        st = model 
        th = copy.deepcopy(st)
        st = TeachingModelWrapper(st, projectors)
        th = TeachingModelWrapper(th, projectors).eval()

        return st, th
