# Helper modules

```python
class: SSLProject.utils.support_set.SupportBuffer(buffer_capacity: int=98304, vectors_dim: int=256)
```

Class for implement base methods support set. As support set using cyclic buffer
- buffer_capacity: number of vectors in buffer. By default: 98304
- vectors_dim: dimensional of vectors in buffer. By default: 256


```python
class: SSLProject.utils.support_set.SupportBufferKNN(buffer_capacity: int=98304, vectors_dim: int=256, k: int=5)
```

Support set class implementation with opportunity find K nearest neighbors in support set. Subclass of 
SSLProject.utils.support_set.SupportBuffer.

- k: number of return nearest neighbors 