

# SSG-LDL (Deprecated)

An unofficial python implementation of SSG-LDL ([González et al. 2021](https://github.com/SpriteMisaka/SSG-LDL/blob/main/bibliography/gonz%C3%A1lez2021.pdf)).

This repo is deprecated. Use [PyLDL](https://github.com/SpriteMisaka/PyLDL) instead.

## Usage

```python
import scipy.io as sio

data = sio.loadmat('SJAFFE')
model = SSG_LDL()
new_data = model.fit_transform(data)
```
