# How to ELMo

Take a look at `elmo_check.py`

## Creating the ELMo model

```python
from elmo import Elmo
elmo = Elmo.get_default(2)
```
Gets an Elmo module that will return elmo embeddings with 2 layers

## Cranking out embeddings

```python
from elmo import Elmo, batch_to_ids
elmo = Elmo.get_default(2)

my_sentences =[["hi", "there"], ["wow", "elmo", "is", "big", "boye"]]
char_indices = batch_to_ids(my_sentences)  # a torch.Tensor
char_indices_cuda = char_indices.cuda()

elmo.cuda()  # Elmo can be placed on the GPU since it is a nn.Module
embeddings = elmo(char_indices_cuda)
```
`embeddings['elmo_representations']` is a length two list of tensors.

Each element contains one layer of ELMo representations with shape `(2, 3, 1024)`.

2    - the batch size
 
3    - the sequence length of the batch

1024 - the length of each ELMo vector

Usually you only need to use ELMo with only 1 output representation, increasing the number
causes elmo to train multiple different mixtures of ELMo representation.

#### Special thanks to AllenNLP from where this code and the trained model was taken :)
