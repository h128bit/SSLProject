from typing import Callable, Tuple
import torch 

from info_nce import InfoNCE

from SSLProject.methods.base import BaseMomentum
from SSLProject.utils import PositionalEncoding, SupportBufferKNN, SupportBuffer, off_diagonal, build_linear_model



class All4One(BaseMomentum):
    def __init__(self, 
                 model: torch.nn.Module, 
                 loss_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]|None=None, 
                 theta: float=0.98, 
                 device: str|None=None,
                 projector_out_size: int=256, 
                 buffer:SupportBuffer|None=None, 
                 temperature: float=0.1, 
                 sigma: float=0.5,
                 kappa: float=0.5,
                 eta: float=5.0,
                 symmetric: bool=True):
        super().__init__(model, loss_func, theta, device)

        self.projector_out_size = projector_out_size
        self.predictor_width = 4096 
        self.sym = symmetric

        self.projector_student = build_linear_model(in_feature_dim=self.model_out_feature, out_feature_dim=self.projector_out_size, middle_feat_layers_dim=self.predictor_width)

        self.projector_teacher = torch.optim.swa_utils.AveragedModel(self.projector_student, avg_fn=self.ema_avg_func)
        
        self.predictor_knn = build_linear_model(in_feature_dim=self.projector_out_size, out_feature_dim=self.projector_out_size, middle_feat_layers_dim=self.predictor_width)

        self.predictor_centroid = build_linear_model(in_feature_dim=self.projector_out_size, out_feature_dim=self.projector_out_size, middle_feat_layers_dim=self.predictor_width)

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.projector_out_size,
            nhead=8,
            dim_feedforward=self.projector_out_size * 2,
            batch_first=True,
            dropout=0.1,
        )

        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=3)

        self.buffer = buffer if buffer else SupportBufferKNN(k=5)

        self.infonce_loss = InfoNCE(temperature=temperature)
        self.pos_encoder = PositionalEncoding(d_model=self.projector_out_size)

        self.a1 = sigma
        self.a2 = kappa
        self.a3 = eta


    def train_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> dict[str, torch.Tensor]:
        view1, view2 = batch

        base_loss = None 
        nnclr_loss = 0
        centroid_loss = 0
        red_loss = 0

        sym_coef = 1 / (self.sym + 1)

        update_buffer = False

        for views in [(view1, view2), (view2, view1)][0:self.sym+1]:
            res_dict = super().train_step(views)
            z_s = res_dict["z_student"]
            z_t = res_dict["z_teacher"]

            if base_loss is None:
                base_loss = res_dict["loss"]
            
            prj_s = self.projector_student(z_s)
            prj_t = self.projector_teacher(z_t)

            pred_s_knn = torch.nn.functional.normalize( self.predictor_knn(prj_s) )
            pred_s_centroid = torch.nn.functional.normalize( self.predictor_centroid(prj_s) )

            kneighbors = self.buffer.find_nn(prj_t) # return normalised vectors

            pos_encod_knn = self.pos_encoder(kneighbors)

            # Shift Operation
            poss_encod_pred_centroid = self.pos_encoder(
                torch.cat((pred_s_centroid.unsqueeze(1), kneighbors), 1)[:, :5, :]
                )

            transf_t = self.transformer_encoder(pos_encod_knn)[:, 0:, :]
            transf_s = self.transformer_encoder(poss_encod_pred_centroid)[:, 0:, :]

            corr_matrix = torch.nn.functional.normalize(prj_s, dim=1).T @ torch.nn.functional.normalize(prj_t, dim=1)

            on_diag_feat = (torch.diagonal(corr_matrix).add(-1).pow(2).mean()).sqrt()
            out_diag_feat = (off_diagonal(corr_matrix).pow(2).mean()).sqrt()

            ## compute losses
            nnclr_loss = nnclr_loss + self.infonce_loss(pred_s_knn, kneighbors[:, 0, :]) 

            centroid_loss = centroid_loss + self.infonce_loss(transf_s[:, 0, :], transf_t[:, 0, :])

            red_loss = red_loss + (0.5 * on_diag_feat + 0.5 * out_diag_feat) * 10

            if not update_buffer:
                self.buffer.put(prj_t)
                update_buffer = True 

        final_loss = base_loss + (self.a1 * nnclr_loss + self.a2 * centroid_loss + self.a3 * red_loss) * sym_coef

        return {
            "base_loss": base_loss,
            "nnclr_loss": nnclr_loss,
            "centroid_loss": centroid_loss,
            "red_loss": red_loss,
            "loss": final_loss
        }
    

    def update_teacher_weights(self) -> None:
        super().update_teacher_weights()
        self.projector_teacher.update_parameters(self.projector_student)