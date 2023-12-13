import torch
import torch.nn as nn


class CombinedSRCNNClassifier(nn.Module):
    def __init__(self, srcnn, classifier):
        super(CombinedSRCNNClassifier, self).__init__()
        self.srcnn = srcnn
        self.classifier = classifier

    @staticmethod
    def reshape_to_model_output(batch, labels, device):
        hgit_res_imgs, true_label = labels
        batch, true_label = batch.permute(1, 0, 2, 3, 4).to(device), true_label.permute(
            1, 0
        ).to(device)
        hgit_res_imgs = hgit_res_imgs.permute(1, 0, 2, 3, 4).to(device)
        return batch, [hgit_res_imgs, true_label]

    def forward(self, inputs):
        batch_size = inputs.size(0)
        x = inputs.view(batch_size * 5, 32, 492, 2)

        rec_img = self.srcnn(x)
        rec_img = rec_img.view(batch_size, 5, 32, 492, 2)
        y_labels_pred = self.classifier(rec_img)

        return rec_img, y_labels_pred
