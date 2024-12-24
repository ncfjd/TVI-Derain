# Textual-Visual Interaction for Enhanced Single Image Deraining using Adapter-Tuned VLMs
## Abstract

This paper proposes a novel method called Textual-Visual Interaction for Enhanced Single Image Deraining using Adapter-Tuned VLMs (TVI-Derain). By leveraging the extensive textual knowledge from pretrained visual-language models (VLMs), we aim to improve the performance of single image deraining. To address the gap between VLMs and the restoration model, we introduce textual-aware intra-layer (TaIl) adapters that adapt the features of downstream data by capturing task-specific knowledge. Furthermore, a textual-visual feature interaction (TVI) module is designed to bridge the gap between textual and visual features, enabling reliable interaction. The proposed cross-attention feature interaction (CAFI) block within the TVI module effectively represents the interactive features. Semantic and degradation textual prompts are integrated as inputs to the text encoder to mitigate semantic disconnection arising from degraded samples. Extensive experimental results on benchmark datasets demonstrate that our method outperforms other competitive methods in terms of performance, showcasing its potential applications in automotive vision systems and surveillance.

![fig1.pdf](https://github.com/ncfjd/TVI-Derain/blob/main/figure/fig.pdf)

## Datasets
* prepare data
  * ```mkdir ./datasets/Rain13k```

  * download the [train](https://drive.google.com/drive/folders/1Hnnlc5kI0v9_BtfMytC2LR5VpLAFZtVe?usp=sharing) set and [test](https://drive.google.com/drive/folders/1PDWggNh8ylevFmrjo-JEvlmqsDlWWvZs?usp=sharing) set 


## Training
The training code will be released after the paper is accepted.
You should change the path to yours in the `Train.py` file.  Then run the following script to test the trained model:

```sh
python Train.py
```

## Testing
You should change the path to yours in the `test.py` file.  Then run the following script to test the trained model:

```sh
python test.py
```
## Notes

1. Send e-mail to 230520854000555@xy.dlpu.edu.cn if you have critical issues to be addressed.
2. Please note that there exists the slight gap in the final version due to errors caused by different testing devices and environments.

## Citations

If TVI-Derain helps your research or work, please consider citing TVI-Derain.
```
@InProceedings{,
  
}
```

## Acknowledgment

This code is based on the [PromptIR](https://github.com/va1shn9v/PromptIR),[DA-CLIP](https://github.com/Algolzw/daclip-uir),[RLP](https://github.com/zkawfanx/RLP). Thanks for their awesome work.
