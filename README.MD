# MetaLion: Truthful Meta Explanations
Truthful Meta-Explanations for Local Interpretability of Machine Learning Models

Automated Machine Learning-based systems' integration into a wide range of tasks has expanded as a result of their performance and speed. Although there are numerous advantages to employing ML-based systems, if they are not interpretable, they should not be used in critical, high-risk applications where human lives are at risk. To address this issue, researchers and businesses have been focusing on finding ways to improve the interpretability of complex ML systems, and several such methods have been developed. Indeed, there are so many developed techniques that it is difficult for practitioners to choose the best among them for their applications, even when using evaluation metrics. As a result, the demand for a selection tool, a meta-explanation technique based on a high-quality evaluation metric, is apparent. In this paper, we present a local meta-explanation technique which builds on top of the truthfulness metric, which is a faithfulness-based metric. We demonstrate the effectiveness of both the technique and the metric by concretely defining all the concepts and through experimentation.

## Based on our previous work: [Altruist / ˈæl tru ɪst /](https://github.com/iamollas/Altruist) 
a person unselfishly concerned for or devoted to the welfare of others (opposed to egoist).

## Instructions
1. Please install the requirements as the can be found in the requirements.txt.zip file
2. Download the trained models, data, preprocessed data and interpretation weights, in each folder. Check the dataset folder and the experiments/quantitative folders

## Contributors on MetaLion
Name | Email
--- | ---
[Ioannis Mollas](https://intelligence.csd.auth.gr/people/ioannis-mollas/) | iamollas@csd.auth.gr
[Grigorios Tsoumakas](https://intelligence.csd.auth.gr/people/tsoumakas/) | greg@csd.auth.gr
[Nick Bassiliades](https://intelligence.csd.auth.gr/people/bassiliades/) | nbassili@csd.auth.gr

## See our Work
[LionLearn Interpretability Library](https://github.com/intelligence-csd-auth-gr/LionLearn) containing: 
1. [LioNets](https://github.com/iamollas/LionLearn/tree/master/LioNets): Local Interpretation Of Neural nETworkS through penultimate layer decoding
2. [LionForests](https://github.com/iamollas/LionLearn/tree/master/LionForests): Local Interpretation Of raNdom FORESts through paTh Selection

## Citation
Please cite our preprint, or our preliminary paper, till this one gets published, if you use it in your work or experiments :D:
```
@misc{mollas2022truthful,
      title={Truthful Meta-Explanations for Local Interpretability of Machine Learning Models}, 
      author={Ioannis Mollas and Nick Bassiliades and Grigorios Tsoumakas},
      year={2022},
      eprint={2212.03513},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}

@inproceedings{mollas2022altruist,
  title={Altruist: argumentative explanations through local interpretations of predictive models},
  author={Mollas, Ioannis and Bassiliades, Nick and Tsoumakas, Grigorios},
  booktitle={Proceedings of the 12th Hellenic Conference on Artificial Intelligence},
  pages={1--10},
  year={2022}
}
```

## License
[GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/)
