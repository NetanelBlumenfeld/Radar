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
        # batch_size = inputs.size(1)
        # x = inputs.reshape(batch_size * 5 * 2, 1, 32, 492)

        # rec_img = self.srcnn(x)
        # rec_img = rec_img.reshape(5, batch_size, 2, 32, 492)
        # y_labels_pred = self.classifier(rec_img)
        # inputs shape: (sequence=5, batch, H, W, channels=2)
        sequence_length, batch_size, channels, H, W = inputs.size()

        # Process each sequence element with self.srcnn
        processed_sequence = []
        for i in range(sequence_length):
            # Extract the sequence element and add a channel dimension
            x = inputs[i].reshape(batch_size * channels, 1, H, W)

            # Apply srcnn
            rec_img = self.srcnn(x)

            # Remove the channel dimension and add it to the processed list
            processed_sequence.append(rec_img.reshape(batch_size, channels, H, W))

        # Recombine the sequence
        rec_img = torch.stack(processed_sequence, dim=0)

        # Apply the classifier
        y_labels_pred = self.classifier(rec_img)
        return rec_img, y_labels_pred
