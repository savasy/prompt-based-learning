# prompt-based-learning

This is a simple implementation of how to leverage a Language Model for a prompt-based learning model.

Prompt-based learning is a new paradigm in the NLP field. In prompt-based learning, we do not have to hold any supervised learning process since we directly rely on the objective function (such as MLM) of any pre-trained language model. 

In order to use the models to achieve prediction tasks, the only thing to be done is to modify the original input<X> using a task-specific template into a textual string prompt such as
  * <X, that is [MASK]> 
 
 
 ```
 prompting= Prompting(model="dbmdz/bert-base-turkish-cased")

text="Çok keyif aldım filmden" # Which means: I liked the film,
propmt=". çünkü [MASK] idi." #  since it was [MASK]
prompted= text + propmt
prompting.prompt_pred(prompted)[:10]
```
> [('güzel', tensor(0.0294)),
 ('eski', tensor(0.0228)),
 ('harika', tensor(0.0220)),
 ('mükemmel', tensor(0.0214)),
 ('eğlenceli', tensor(0.0209)),
 ('yeni', tensor(0.0204)),
 ('muhteşem', tensor(0.0184)),
 ('kötü', tensor(0.0179)),
 ('komik', tensor(0.0171)),
 ('iyi', tensor(0.0169))]

 
 
 so that the model can achieve the task even without learning.

  Such a mechanism allows us to exploit the LM that is pre-trained on huge amounts of textual data. This prompting function can be defined to make any LM be able to achieve few-shot, one-shot, or even zero-shot learning tasks where we easily adapt the model to new scenarios even with few or no labeled data.

