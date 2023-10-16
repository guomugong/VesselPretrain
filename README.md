## Bezier Curves Meet Deep Learning: A Novel Pretraining Method for Improved and Generalized Retinal Vessel Segmentation
Please read our [paper](https://xxx) for more details!
### Introduction:
Accurate segmentation of retinal vessels plays a crucial role in the computer-aided diagnosis of ocular diseases. In recent times, deep learning techniques have been employed for vessel segmentation, yielding promising outcomes. Nonetheless, two significant issues remain unresolved. Firstly, the weights of the majority of deep learning models are typically initialized randomly or transferred from pretrained ImageNet models, potentially leading to suboptimal segmentation performance.  Secondly, the scarcity of annotated fundus images impedes the generalization ability of deep learning models. Motivated by the notion that vessel structures can be simulated using B´ezier curves, this paper introduces an image synthesis approach to tackle these challenges. Specifically, we generate a multitude of vessel structures using B´ezier curves, employing these synthesized images for pretraining a vessel segmentation model. Subsequently, fine-tuning of the model is performed on the dedicated vessel segmentation dataset. Experimental results demonstrate a noteworthy enhancement in our method’s F1-score compared to random initialization. Specifically, our approach yields improvements of 1.8%, 4.7%, and 0.5% on the DRIVE, STARE, and CHASE_DB1 datasets, respectively. Furthermore, when subjected to cross-dataset evaluation, our method achieves an F1-score improvement exceeding 10.0%, underscoring its robust generalization capability.


# Training
1. Download the [DDR](https://github.com/nkicsl/OIA) dataset.
2. Run generate_ddr.py to generate synthesized images.
3. Run pretrain.py for pretraining on the synthesized images.
4. Run train.py for fine-tuning on the vessel segmentation datasets.
