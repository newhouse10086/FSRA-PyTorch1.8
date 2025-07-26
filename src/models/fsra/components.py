"""FSRA model components and utilities."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init_kaiming(m):
    """Kaiming initialization for weights."""
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    """Classifier initialization for weights."""
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class GeM(nn.Module):
    """Generalized Mean Pooling with learnable parameter."""
    
    def __init__(self, dim=768, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(dim) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        """Generalized mean pooling."""
        p = F.softmax(p, dim=0).unsqueeze(-1)
        x = torch.matmul(x, p)
        x = x.view(x.size(0), x.size(1))
        return x


class ClassBlock(nn.Module):
    """Classification block with optional bottleneck."""
    
    def __init__(self, input_dim, class_num, droprate=0.5, relu=False, bnorm=True, 
                 num_bottleneck=512, linear=True, return_f=False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
            
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate > 0:
            add_block += [nn.Dropout(p=droprate)]
            
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return [x, f]
        else:
            x = self.classifier(x)
            return x


class FeatureBlock(nn.Module):
    """Feature extraction block."""
    
    def __init__(self, input_dim, num_bottleneck=512, add_l2norm=True, has_dropout=True, 
                 has_bn=True, has_relu=True, num_classes=0):
        super(FeatureBlock, self).__init__()
        self.add_l2norm = add_l2norm
        self.has_dropout = has_dropout
        self.num_classes = num_classes
        
        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)]
        
        if has_bn:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if has_relu:
            add_block += [nn.LeakyReLU(0.1)]
        if has_dropout:
            add_block += [nn.Dropout(p=0.5)]
            
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        if num_classes > 0:
            classifier += [nn.Linear(num_bottleneck, num_classes)]
            classifier = nn.Sequential(*classifier)
            classifier.apply(weights_init_classifier)
        else:
            classifier = None

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        
        if self.add_l2norm:
            x = F.normalize(x, p=2, dim=1)
            
        if self.classifier is not None:
            y = self.classifier(x)
            return y, x
        else:
            return x


class MultiScaleBlock(nn.Module):
    """Multi-scale feature extraction block."""
    
    def __init__(self, input_dim, num_classes, num_bottleneck=512):
        super(MultiScaleBlock, self).__init__()
        
        self.num_classes = num_classes
        self.num_bottleneck = num_bottleneck
        
        # Global feature
        self.global_block = ClassBlock(
            input_dim, num_classes, droprate=0.5, relu=False, bnorm=True,
            num_bottleneck=num_bottleneck, linear=True, return_f=True)
        
        # Local features
        self.local_blocks = nn.ModuleList([
            ClassBlock(
                input_dim, num_classes, droprate=0.5, relu=False, bnorm=True,
                num_bottleneck=num_bottleneck, linear=True, return_f=True)
            for _ in range(4)  # 4 local regions
        ])

    def forward(self, global_feat, local_feats):
        """
        Args:
            global_feat: Global feature tensor
            local_feats: List of local feature tensors
        """
        # Global classification
        global_out = self.global_block(global_feat)
        
        # Local classifications
        local_outs = []
        for i, local_feat in enumerate(local_feats):
            if i < len(self.local_blocks):
                local_out = self.local_blocks[i](local_feat)
                local_outs.append(local_out)
        
        return global_out, local_outs
