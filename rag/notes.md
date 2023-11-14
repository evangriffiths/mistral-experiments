# RAG

1. Chunk 'extra context' data sources into reasonable, same-sized sequences.
2. Embed these, and store in a vector DB.
3. To fetch some extra context before passing a query to an LLM, embed the original query.
4. Look up top-k most similar context chunks from the DB using cosine similarity (normalized dot-product) with the query embedding.
5. Augment the original query with the extra context and pass to the LLM.

## Embeddings

A naiive implementation of computing a sequence embedding might be to take counts of words in the sequence, and normalize. But this has 2 big limitations:

- It doesn't take into account 'similarity of meaning'. e.g. you want 'queen' and 'monarch' to be close together in embedding space.
- Tokenizing at the word level requires very large vectors due to a very large vocab size.

We can take our learnings on embeddings from before (see `notes.md` in `makemore`), and apply them here: The first layer in a GPT is an embedding layer. This converts the sequence representation from a vector of sub-word token indices of length `T` to a `(T, C)` matrix of token embeddings. In this higher dimensional feature space, tokens are given a better semantic representation. Passing these embeddings through transformer layers gives them additional contextual meaning. SOTA embedding models are *similar* to this, but not quite the same.

Q: Why not use the same GPT backbone to compute the sequence embeddings for RAG? A: Because GPTs contain decoder-only transformer layers. This means each embedding only has context of previous tokens, so less powerful!

In practice embedding models use BERT-like backbones, which contain encoder-based transformer layers.

So this explains how we compute *token embeddings*. But how do we aggregate these to get *sequence embeddings*? In the BERT model case, we actually get this aggregation for free. This is because sequences fed to the BERT model during training always start with a special token, `[CLS]`. The embedding for this single token can be used as a proxy for the sequence embedding. To understand how this works, we need to consider how the BERT model is trained.

Pretraining consists of 2 stages: Masked language modelling (MLM) and Next sentence prediction (NSP). In NSP, pairs of sentences are sampled, and the model is trained to predict whether the second sentence in the pair follows the first in the original text or not. The input sequence is made from the two sentence + special tokens: `[CLS]` at the beginning, `[SEP]` to separate between them. The final binary classification layer takes the embedding of the `[CLS]` token as input, and produces a probability of the sentences being valid pairs. This is combined with the target to produce the loss.

Through this process, the model learns to embed the `[CLS]` token with a useful representation of the whole sentence, so can be used as an embedding model for a downstream RAG application.

### RAG vs Finetuning

- If your knowledge base changes frequently, then it's easier to stay up to date using RAG. It's easier to refresh the vector DB vs. making another finetuning run.
- RAG is better if you want to be able to quote the source of information.
- A finetuned model will often give better quality responses. Filling your prompt with large chunks of extra context from your RAG DB is bad for:
  - cost (if using an API that charges per input token)
  - chance of forgetting/ignoring parts of the prompt in the output response.
- Tradeoffs w.r.t infrastructure and technical expertise. If you don't have access to weights of model you're using (e.g. if you're just using an API) then you can't fine tune!
