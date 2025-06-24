| Model                                              | Test Set AP50 |
| :------------------------------------------------- | :-----------: |
| **Anchor-Free (AF) Methods**                       |               |
| FSAF                                               |     0.701     |
| ATSS                                               |     0.692     |
| FoveaBox                                           |     0.692     |
| VarifocalNet (2)                                   |     0.683     |
| VarifocalNet (1)                                   |     0.664     |
| **One-Stage Methods**                              |               |
| YOLO-MS (Epoch 200)                                |     0.748     |
| YOLO-MS (Epoch 300)                                |     0.739     |
| Gradient Harmonized Single-stage Detector          |     0.691     |
| Generalized Focal Loss                             |     0.677     |
| Probabilistic Anchor Assignment                    |     0.677     |
| SABL                                               |     0.661     |
| NAS-FPN                                            |     0.658     |
| RetinaNet                                          |     0.650     |
| YoloV3                                             |     0.591     |
| **Two-Stage and Multi-Stage (DetectorRS) Methods** |               |
| Double Heads                                       |     0.699     |
| CARAFE                                             |     0.697     |
| Empirical Attention                                |     0.690     |
| Mixed precision training                           |     0.679     |
| Faster R-CNN                                       |     0.660     |
| Deformable ConvNets v2                             |     0.657     |
| Dynamic R-CNN                                      |     0.655     |
| DetecorRS                                          |     0.651     |
| Weight Standardization                             |     0.631     |

| model     |  mAP  | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
| :-------- | :---: | :----: | :----: | :---: | :---: | :---: |
| Epoch 300 | 0.405 | 0.739  | 0.403  | 0.227 | 0.449 | 0.653 |
| Epoch 200 | 0.416 | 0.748  | 0.421  | 0.246 | 0.463 | 0.616 |
