from typing import Callable
import torch 

from info_nce import InfoNCE

from SSLProject.methods.base import BaseMomentum
from SSLProject.support_set import SupportBufferKNN, SupportBuffer

from SSLProject.utils import PositionalEncoding, off_diagonal
from SSLProject.methods.factory.model_builder import build_linear_model, TeacherStudentBuilder



class All4One(BaseMomentum):
    def __init__(self, 
                 model: torch.nn.Module, 
                 loss_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]|None=None, 
                 theta: float=0.98,
                 projector_out_size: int=256, 
                 predictor_width: int|list[int]=4096,
                 buffer: SupportBuffer|None=None, 
                 temperature: float=0.1,
                 alpha: float=1, 
                 sigma: float=0.5,
                 kappa: float=0.5,
                 eta: float=5.0,
                 T: float=0.1,
                 symmetric: bool=True) -> None:
        
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=projector_out_size,
            nhead=8,
            dim_feedforward=projector_out_size * 2,
            batch_first=True,
            dropout=0.1,
        )

        projectors = [
            ("projector", build_linear_model(in_feature_dim=model.out_features, out_feature_dim=projector_out_size, middle_feat_layers_dim=predictor_width)),
            ("predictor_knn", build_linear_model(in_feature_dim=projector_out_size, out_feature_dim=projector_out_size, middle_feat_layers_dim=predictor_width)),
            ("predictor_centroid", build_linear_model(in_feature_dim=projector_out_size, out_feature_dim=projector_out_size, middle_feat_layers_dim=predictor_width)),
            ("transformer_encoder", torch.nn.TransformerEncoder(encoder_layer, num_layers=3))
        ]
        student, teacher = TeacherStudentBuilder.build(model, projectors=projectors)

        super().__init__(student, teacher, loss_func, T, theta)

        self.projector_out_size = projector_out_size
        self.predictor_width = predictor_width 
        self.sym = symmetric

        self.buffer = buffer if buffer else SupportBufferKNN(k=5)

        self.infonce_loss = InfoNCE(temperature=temperature)
        self.pos_encoder = PositionalEncoding(d_model=self.projector_out_size)

        self.alpha = alpha
        self.a1 = sigma
        self.a2 = kappa
        self.a3 = eta

        for param in self.teacher.parameters():
            param.requires_grad = False
        for param in self.buffer.parameters():
            param.requires_grad = False


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

            print(f"STUDENT OUT SHAPE {res_dict["student_out"].shape}")
            print(f"TEACHER OUT SHAPE {res_dict["teacher_out"].shape}")

            prj_s = self.student.projector(res_dict["student_out"])
            prj_t = self.teacher.projector(res_dict["teacher_out"])

            if base_loss is None:
                base_loss = res_dict["loss"]

            pred_s_knn = torch.nn.functional.normalize( self.student.predictor_knn(prj_s) )
            pred_s_centroid = torch.nn.functional.normalize( self.student.predictor_centroid(prj_s) )

            kneighbors = self.buffer.find_nn(prj_t) # return normalised vectors

            pos_encod_knn = self.pos_encoder(kneighbors)

            # Shift Operation
            poss_encod_pred_centroid = self.pos_encoder(
                torch.cat((pred_s_centroid.unsqueeze(1), kneighbors), 1)[:, :5, :]
                )

            transf_t = self.student.transformer_encoder(pos_encod_knn)[:, 0:, :]
            transf_s = self.student.transformer_encoder(poss_encod_pred_centroid)[:, 0:, :]

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

        final_loss = self.alpha * base_loss + (self.a1 * nnclr_loss + self.a2 * centroid_loss + self.a3 * red_loss) * sym_coef

        return {
            "base_loss": base_loss,
            "nnclr_loss": nnclr_loss,
            "centroid_loss": centroid_loss,
            "red_loss": red_loss,
            "loss": final_loss
        }
    
    def forward(self, x) -> dict:
        return self.train_step(x)
