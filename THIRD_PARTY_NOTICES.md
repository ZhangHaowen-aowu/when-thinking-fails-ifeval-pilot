# Third-party attribution and license notes

- `instruction_following_eval/*.py` contains a copy of the official Google Research IFEval evaluator implementation: https://github.com/google-research/google-research/tree/master/instruction_following_eval. These files include their upstream Apache-2.0 source headers. The upstream Apache-2.0 license text is included at `instruction_following_eval/LICENSE` for convenience. The precise upstream git commit used when the local files were downloaded was not recorded.
- `input_data.jsonl` and the 60-prompt subset originate from the [Google IFEval benchmark](https://huggingface.co/datasets/google/IFEval) (Zhou et al., 2023). Credit and any dataset license conditions should be preserved.
- The CoT prompt in `run_experiment.py` / `run_experiment_v2.py` follows the published prompt template in the authors' [When Thinking Fails source](https://github.com/amazon-science/when-thinking-fails-RLLM-if-evaluation).
- The model output was obtained via a hosted Qwen2.5-7B-Instruct model; publication/reuse of model generations should comply with provider terms and any applicable benchmark restrictions.
- No blanket license is granted for all files by this notice; repository owners should select a license for their independently authored code if they intend reuse.
