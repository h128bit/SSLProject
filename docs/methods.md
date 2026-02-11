# Self supervised learning methods API documentation

## Base classes

```python
class: SSLProject.methods.base.BaseMethod(model: torch.nn.Module, loss_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]|None=None,
device: str|None=None)
```
Base abstract class


```python
class: SSLProject.methods.base.BaseMomentum(model: torch.nn.Module, loss_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]|None=None,
theta: float=0.98, device: str|None=None)
```
Subclass of BaseMethod. Implement update teacher weights by Exponential Moving Average formula; 
compute loss between teacher out as target and student out as predict.

---

## Methods

```python
class: SSLProject.methods.all4one.All4One(model: torch.nn.Module, loss_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]|None=None, 
theta: float=0.98, device: str|None=None, projector_out_size: int=256, buffer:SupportBuffer|None=None, 
temperature: float=0.1, sigma: float=0.5, kappa: float=0.5, eta: float=5.0, symmetric: bool=True)
```

Implementation of All4One method.

- model: Pytorch model. Model class must have parametr <code>out_features</code>
- loss_func: used for compute loss between teacher outputs as true labels and student outputs as predict.
By default: torch.nn.functional.cross_entropy
- theta: coeficient for Exponential Moving Average. By default: 0.98
- device: device for locate model. If set to None used allowed device. By default None 
- projector_out_size: dimensional of output vector of projector. By default: 256
- buffer: implementation of support set. If set in None when will use SSLProject.util.support_set.SupportBufferKNN.
By default: None
- temperature: temperature for InfoNCE loss
- sigma: coefficient for nnclr loss
- kappa: coefficient for centroid loss
- eta: coefficient for red loss
- symmetric: if set to True when both view of augmented image one by one pass to teacher and student model.
If set to False when only first view pass in teacher and student model.
*pseudocode*
~~~python
for views in [(view1, view2), (view2, view1)][0:self.sym+1]:
    res_dict = super().train_step(views)
    t_out = teacher(views[0])
    s_out = student(views[1])
    loss = compute_loss(s_out, t_out)
~~~


### Methods

```python
train_step(batch: tuple[torch.Tensor, torch.Tensor])
```
Train step on batch. In batch first veiw pass to student, second view pas to teacher.
Compute loss as 
<code>final_loss = base_loss + (sigma * nnclr_loss + kappa * centroid_loss + eta * red_loss) * sym_coef</code>,
where 
- base_loss are compute between teacher out as true labels and student out as predict
- sym_coef is 1 if <code>symmetric</code> set to False and 0.5 if <code>symmetric</code> set to True

**return**
dict of losses with keys
- base_loss
- nnclr_loss
- centroid_loss 
- red_loss
- loss (is final_loss)