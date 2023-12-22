import torch
from network.models.basic_model import BasicModel


class CombinedSRCNNClassifier(BasicModel):
    def __init__(
        self,
        srcnn: BasicModel,
        classifier: BasicModel,
        scale_factor: int = 1,
        only_wights: bool = False,
    ):
        model_name = f"sr_{srcnn.model_name}_classifier_{classifier.model_name}"
        super(CombinedSRCNNClassifier, self).__init__(model_name, only_wights)
        self.srcnn = srcnn
        self.classifier = classifier
        self.scale_factor = scale_factor

    @staticmethod
    def reshape_to_model_output(batch, labels, device):
        hgit_res_imgs, true_label = labels
        batch, true_label = batch.permute(1, 0, 2, 3, 4).to(device), true_label.permute(
            1, 0
        ).to(device)
        hgit_res_imgs = hgit_res_imgs.permute(1, 0, 2, 3, 4).to(device)
        return batch, [hgit_res_imgs, true_label]

    def forward(self, inputs):
        sequence_length, batch_size, channels, H, W = inputs.size()

        # Process each sequence element with self.srcnn
        processed_sequence = []
        for i in range(sequence_length):
            # Extract the sequence element and add a channel dimension
            x = inputs[i].reshape(batch_size * channels, 1, H, W)

            # Apply srcnn
            rec_img = self.srcnn(x)

            # Remove the channel dimension and add it to the processed list
            processed_sequence.append(
                rec_img.reshape(
                    batch_size, channels, H * self.scale_factor, W * self.scale_factor
                )
            )

        # Recombine the sequence
        rec_img = torch.stack(processed_sequence, dim=0)

        # Apply the classifier
        y_labels_pred = self.classifier(rec_img)
        return rec_img, y_labels_pred
