
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import QResNetConfig
from .blocks import ClassicalResBlock
from .quantum_layer import QuantumLayer

class QResNet(nn.Module):
    def __init__(self, cfg: QResNetConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or QResNetConfig()
        self.conv1 = nn.Conv2d(1, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = nn.Sequential(ClassicalResBlock(16,16,1), ClassicalResBlock(16,16,1))
        self.layer2 = nn.Sequential(ClassicalResBlock(16,32,2), ClassicalResBlock(32,32,1))
        self.layer3 = nn.Sequential(ClassicalResBlock(32,64,2), ClassicalResBlock(64,64,1))
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.qmap = nn.Linear(64, self.cfg.n_qubits)
        self.ql = QuantumLayer(self.cfg.n_qubits, self.cfg.n_quantum_layers, self.cfg.shots, self.cfg.use_quantum_eval)
        self.fc1 = nn.Linear(self.cfg.n_qubits, 32)
        self.do = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, self.cfg.num_classes)
        self._init()

    def _init(self)->None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0); nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=5**0.5); nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out); out = self.layer2(out); out = self.layer3(out)
        out = self.pool(out).view(out.size(0), -1)
        q_in = self.qmap(out)
        q_out = self.ql(q_in)
        out = F.relu(self.fc1(q_out)); out = self.do(out); out = self.fc2(out)
        return out
