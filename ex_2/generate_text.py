from aitextgen import aitextgen

if __name__ == '__main__':
    out_dir = "../results/"

    ai = aitextgen(model_folder=out_dir, tokenizer_file="aitextgen.tokenizer.json", verbose=True)

    ai.generate_to_file(n=10, prompt="Twitter ", max_length=100, temperature=1.2)
    # ai.generate_to_file(
    #     prompt="Twitter ",
    #     seed=27,
    #     # Model params
    #     n=10,
    #     min_len=None,
    #     max_len=256,
    #     temperature=1.2,
    #     do_sample=True,
    #     use_cache=True,
    #     # Custom model params
    #     early_stopping=False,  # whether to stop beam search when at least num_beams sentences are finished
    #     num_beams=1,  # num beams for beam search, 1 = no beam search
    #     top_k=50,  # num highest probaba tokens to keep for top-k filtering
    #     top_p=0.95,  # float < 1 if most probable tokens with probs that add up to top_p are kept for generation
    #     repetition_penalty=1.2,  # penalty for repetition. 1.0 = no penalty
    #     length_penalty=1.0,  # < 1.0 shorter, > 1.0 longer
    #     no_repeat_ngram_size=0,  # > 0, all ngrams of that size can only occur once.
    #     num_beam_groups=1,  # num groups to divide num_beams into to ensure diversity
    #     diversity_penalty=0.0,  # value subtracted from beamscore if generates token same as any beam from other group
    #     remove_invalid_values=True,
    #     # output
    #     # return_as_list=True,
    #     # lstrip=False,
    #     # skip_special_tokens=False,
    # )


