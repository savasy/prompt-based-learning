# prompt-based-learning

This is a simple implementation of how to leverage a Language Model for a prompt-based learning model.

Prompt-based learning is a new paradigm in the NLP field. In prompt-based learning, we do not have to hold any supervised learning process since we directly rely on the objective function (such as MLM) of any pre-trained language model. 

In order to use the models to achieve prediction tasks, the only thing to be done is to modify the original input<X> using a task-specific template into a textual string prompt such as
  * <X, that is [MASK].> 
 
 Here is a quick run!
 
 ```
# load Prompting class
from prompt import Prompting
prompting= Prompting(model=model_path)
prompt="Because it was [MASK]."
text="I really like the film a lot."
prompting.prompt_pred(text+prompt)[:10]
```
> [('great', tensor(9.5558)),
 ('amazing', tensor(9.2532)),
 ('good', tensor(9.1464)),
 ('fun', tensor(8.3979)),
 ('fantastic', tensor(8.3277)),
 ('wonderful', tensor(8.2719)),
 ('beautiful', tensor(8.1584)),
 ('awesome', tensor(8.1071)),
 ('incredible', tensor(8.0140)),
 ('funny', tensor(7.8785))]

 
Zero-shot performance on IMDB dataset - randomly selected 200 pos 200 neg example
 

model | zero-shot acc.
---|---
bert-base-uncased|73.5
bert-large-uncased | 77.25
 
