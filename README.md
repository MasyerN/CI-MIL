# CI-MIL


In the realm of Whole Slide Image (WSI) classification under deep learning, Multiple Instance Learning (MIL) has garnered widespread attention due to its training process that requires only reported diagnostic results as labels, eliminating the need for manual pixel-wise annotation. Previous MIL research has primarily focused on enhancing feature aggregators for globally analyzing WSIs. However, these methods overlook a crucial diagnostic logic: the prediction of model should be drawn only from areas of the image that contain diagnostic evidence (such as tumor cells), which usually occupy relatively small areas. To establish the true correlation between model predictions and diagnostic evidence subregions, we have developed Causal Inference Multiple Instance Learning (CI-MIL). CI-MIL incorporates feature distillation and causal inference, employing a two-stage approach to select patches with high diagnostic value, thereby stabilizing the genuine causal relationship between model predictions and identifiable diagnostic areas. Initially, through feature distillation, CI-MIL selects patches with a high probability of containing tumor cells. Subsequently, the image features of these patches are mapped to random Fourier space to obtain weighted scores that minimize feature correlation. Using these weighted scores, patch-level features are fused into WSI-level features, ultimately completing the WSI classification. Throughout this process, the feature correlation among homogeneous patches is reduced, and the prediction results exhibit a stronger correlation with the fewer patches that possess clear diagnostic significance, making the prediction more direct and reliable. This method has surpassed current state-of-the-art methods, achieving 92.25% accuracy and 95.28% AUC on Camelyon16 (breast cancer) and 93.81% accuracy and 98.02% AUC on TCGA-NSCLC (non-small cell lung cancer). Additionally, the areas selected by CI-MIL exhibit a high level of consistency with ground truth, showcasing outstanding performance and interpretability.
