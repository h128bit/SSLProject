# SSLProject

## Repository with implementation of self supervised learning methods

---

#### Implemented

- [x] All4One
- [ ] MoCo v3
- [ ] DINO
- [ ] BYOL

---

Repository containse implementation of self supervised learning methods with supported custom base model.
For more details see docs/

## Usage example

```python
import torch
import timm
from pathlib import Path
from PIL import Image
import albumentations as A
import numpy as np

from SSLProject.methods.all4one import All4One
from SSLProject.trainers.simple_trainer import SimpleTrainer


class SSLDataset(torch.utils.data.Dataset):
    def __init__(self, files):
        self.files = files

        self.student_augment = A.Compose([...])

        self.teacher_augment = A.Compose([...])


    def __len__(self):
        return len(self.files)
  

    def __getitem__(self, index):
        image = Image.open(self.files[index]).convert("RGB")
        image = np.array(image)

        student_view = self.student_augment(image=image)["image"]
        teacher_view = self.teacher_augment(image=image)["image"]

        return student_view, teacher_view


class ModelWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
      
        self.model = timm.create_model("resnet18", pretrained=False)
        self.out_features = self.model.fc.out_features

    def forward(self, x):
        return self.model(x)


files = ...

dataset = SSLDataset(files)
dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=16, shuffle=True)

model = ModelWrapper()
method = All4One(model=model)
optimizer = torch.optim.AdamW(method.parameters())
sheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=10)

trainer = SimpleTrainer(method=method,
                        optimizer=optimizer,
                        dataloader=dataloader,
                        sheduler=sheduler,
                        num_epoch=5,
                        project_root_or_url="localhost:5000",
                        logger="mlflow")


trainer.train()
```
