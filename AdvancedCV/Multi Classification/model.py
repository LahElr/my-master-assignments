import utils
import torch
from transformers import VanForImageClassification
from sklearn import metrics
import timm
from timm.loss import LabelSmoothingCrossEntropy
import re

optimizer_index = {
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD,
    "adagrad": torch.optim.Adagrad,
    "adamw": torch.optim.AdamW
}


class VanForImageMultiClassification(torch.nn.Module):
    def __init__(self, scale: str = "base"):
        super().__init__()
        if not scale in ["base", "small", "tiny", "large"]:
            raise ValueError(
                f"model.VanForImageMultiClassification.__init__: scale {scale} unknown."
            )
        self.mainbone = VanForImageClassification.from_pretrained(
            f"Visual-Attention-Network/van-{scale}")
        # base scale model on its own need about 0.2G VRAM
        # home/.cache/huggingface/hub
        # input: pixel_values = tensor torch.Size([bs, 3, 224, 224])
        # output.loss
        # output.logits = tensor torch.Size([bs, output_size])
        output_size = {"base": 512, "tiny": 256, "large":512}[scale]

        # set the last van stage to be trainable
        # "van\\.encoder\\.stages\\.3\\.layers\\.1.*"
        # (van\\.encoder\\.stages\\.3\\.layers\\.2.*)|(van\\.encoder\\.stages\\.3\\.normalization.*)|(van\\.layernorm.*)
        unfreeze_re = utils.read_config("model.unfreeze_re")
        unfreeze_re = re.compile(unfreeze_re)
        for pname, p in self.mainbone.named_parameters():
            if unfreeze_re.match(pname) is not None:
                # if utils.read_config("model.unfreeze_re") in pname:
                p.requires_grad = True
            else:
                p.requires_grad = False

        # delete the original readout
        self.mainbone.classifier = torch.nn.Identity()

        self.dropout = torch.nn.Dropout(p=utils.read_config("model.dropout"))

        self.readout_layers = torch.nn.ModuleList([
            torch.nn.Linear(output_size, _)
            for _ in utils.read_config("model.group_dim")
        ])

        utils.logger.info(
            f"Model init completed. Overall parameter number {sum(p.numel() for p in self.parameters())}, trainable parameter nmber {sum(p.numel() for p in self.parameters() if p.requires_grad)}"
        )

    def forward(self, x):
        output = self.mainbone(pixel_values=x)
        logits = output.logits
        logits = self.dropout(logits)
        preds = [
            readout_layer(logits) for readout_layer in self.readout_layers
        ]
        return preds


class MultiCrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        if utils.read_config("train.label_smoothing") is not None:
            self.loss = LabelSmoothingCrossEntropy(
                smoothing=utils.read_config("train.label_smoothing")
            )  #! note that the bs axis is necessary
        else:
            self.loss = torch.nn.CrossEntropyLoss()
        self.class_weights = utils.read_config("model.group_weights")

    def forward(self, preds, labels):
        '''
        preds: [6 tensors of [bs,?]]
        labels: torch.Size([bs, 6])
        '''
        loss = 0.0
        for i in range(6):
            loss += self.class_weights[i] * self.loss(preds[i], labels[:, i])
        return loss


def calculate_acc(preds, labels):
    '''
    preds: [6 tensors of [bs,?]]
    labels: torch.Size([bs, 6])
    '''
    pred_labels = [_.argmax(1) for _ in preds]
    accs = [0 for _ in range(6)]
    for i in range(6):
        label = labels[:, i]
        pred_label = pred_labels[i]
        accs[i] = metrics.accuracy_score(label.tolist(), pred_label.tolist())
    return accs
