from torch import nn


def linear_block(in_feat, out_feat) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_feat, out_feat),
        nn.BatchNorm1d(out_feat),
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

    