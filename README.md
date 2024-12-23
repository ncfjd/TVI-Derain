# Textual-Visual Interaction for Enhanced Single Image Deraining using Adapter-Tuned VLMs
## Abstract

This paper proposes a novel method called Textual-Visual Interaction for Enhanced Single Image Deraining using Adapter-Tuned VLMs (TVI-Derain). By leveraging the extensive textual knowledge from pretrained visual-language models (VLMs), we aim to improve the performance of single image deraining. To address the gap between VLMs and the restoration model, we introduce textual-aware intra-layer (TaIl) adapters that adapt the features of downstream data by capturing task-specific knowledge. Furthermore, a textual-visual feature interaction (TVI) module is designed to bridge the gap between textual and visual features, enabling reliable interaction. The proposed cross-attention feature interaction (CAFI) block within the TVI module effectively represents the interactive features. Semantic and degradation textual prompts are integrated as inputs to the text encoder to mitigate semantic disconnection arising from degraded samples. Extensive experimental results on benchmark datasets demonstrate that our method outperforms other competitive methods in terms of performance, showcasing its potential applications in automotive vision systems and surveillance.

![20231020101639](https://github.com/ncfjd/TVI-Derain/blob/main/figure/fig.pdf)

## Datasets
<table>
<thead>
  <tr>
    <th>Dataset</th>
    <th>Rain200<H/th>
    <th>Rain200L</th>
    <th>Rain800</th>
    <th>Rain1400</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Baidu Cloud</td>
    <td> <a href="https://pan.baidu.com/disk/main#/index?category=all&path=%2FRain100H_">Download</a> </td>
    <td> <a href="https://pan.baidu.com/disk/main#/index?category=all&path=%2FRain200L ">Download</a> </td>
    <td> <a href="https://pan.baidu.com/disk/main#/index?category=all&path=%2FRain800 ">Download</a> </td>
    <td> <a href="https://pan.baidu.com/disk/main#/index?category=all&path=%2FRain1400 ">Download</a> </td>
  </tr>
</tbody>
</table>
Here, we provide complete datasets, including "Rain200H", "Rain200L", "Rain800" and "Rain1400", and they are fully paired images. 

### Training
The training code will be released after the paper is accepted.
You should change the path to yours in the `Train.py` file.  Then run the following script to test the trained model:

```sh
python Train.py
```

### Testing
You should change the path to yours in the `test.py` file.  Then run the following script to test the trained model:

```sh
python test.py
```
## Notes

1. Send e-mail to 230520854000555@xy.dlpu.edu.cn if you have critical issues to be addressed.
2. Please note that there exists the slight gap in the final version due to errors caused by different testing devices and environments.

## Acknowledgment

This code is based on the [Restormer](https://github.com/swz30/Restormer). Thanks for their awesome work.
