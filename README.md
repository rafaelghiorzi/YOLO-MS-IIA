# Projeto YOLO-MS-IIA

Esse projeto tem como objetivo treinar um modelo YOLO-MS para detecção de árvores em imagens aéreas. O modelo é treinado com base no dataset COCO, mas adaptado para detectar apenas uma classe: árvores.

O objetivo é criar um modelo para comparação com outros vários modelos de detecção de objetos que fazem parte de um estudo conduzido por Pedro Zamboni nesse [link](https://www.mdpi.com/2072-4292/13/13/2482).

O projeto, assim como sua explicação, pode ser encontrada no notebook do Jupyter [aqui](main.ipynb).

Por motivos de armazenamento do GitHub, os arquivos dos modelos treinados não estão presentes na pasta `finetune`. O modelo treinado do zero pode ser encontrado aqui: [yoloms_syncbn_fast_8xb32-300e_coco-0d9c8664-878989a7.pth](https://drive.google.com/file/d/1gX4WxPGVYTM2wdLn49mPzqLSH8lXMv4x/view).O checkpoint com melhor desempenho mAP50 pode ser encontrado aqui: [best_epoch_200.pth](https://drive.google.com/file/d/1bIhDVWrNjNE-8L2GLuYAODsYVihvsGWY/view?usp=sharing). Outros modelos também podem ser encontrados no [Model Zoo](mmyolo/model_zoos.md). Os checkpoints das outras épocas do modelo treinado podem ser obtidas realizando o treinamento novamente, como exemplificado no notebook.
