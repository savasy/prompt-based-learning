# prompt-based-learning

This is a simple implementation of how to leverage a Language Model for a prompt-based learning model.

Prompt-based learning is a new paradigm in the NLP field. In prompt-based learning, we do not have to hold any supervised learning process since we directly rely on the objective function (such as MLM) of any pre-trained language model. 

In order to use the models to achieve prediction tasks, the only thing to be done is to modify the original input<X> using a task-specific template into a textual string prompt such as
  * <X, that is [MASK]> 
 
 so that the model can achieve the task even without learning.

  Such a mechanism allows us to exploit the LM that is pre-trained on huge amounts of textual data. This prompting function can be defined to make any LM be able to achieve few-shot, one-shot, or even zero-shot learning tasks where we easily adapt the model to new scenarios even with few or no labeled data.

