import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from enum import Enum
from typing import Callable, Any, Tuple
from adversarial_attack import AdversarialAttack()

class DefenseType(Enum):
    DISTILLATION = 0
    TRADES = 1,
    MART = 2,
    BAYESIAN = 3,

class Defenses:
    def __init__(self,
                 beta: float = 6.0, # used in TRADES defense
                 eps: float = 8/255, 
                 alpha: float = 2/255, 
                 iters: int = 10,
                 device: torch.device | None = None,
                 teacher_model: torch.nn.Module | None = None, # used in defensive distillation
                 temperature: float = 20,
                ) -> None:
        self.beta = beta
        self.eps = eps
        self.alphs = alpha
        self.iters = iters
        self.device = device
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.teacher_model = teacher_model
        self.temperature = temperature
        
    def pgd_attack(self, 
                   model: torch.nn.Module, 
                   x: torch.tensor, 
                   y: torch.tensor) -> torch.tensor:
        x_adv = x.detach() + torch.zeros_like(x).uniform_(-self.eps, self.eps)
        x_adv = torch.clamp(x_adv, 0, 1)
    
        for _ in range(self.iters):
            x_adv.requires_grad_()
    
            logits = model(x_adv)
            loss = F.cross_entropy(logits, y)
    
            grad = torch.autograd.grad(loss, x_adv)[0]
    
            x_adv = x_adv.detach() + self.alpha * torch.sign(grad)
            x_adv = torch.min(torch.max(x_adv, x - self.eps), x + self.eps)
            x_adv = torch.clamp(x_adv, 0, 1)
    
        return x_adv

    def adversarial_training(self, 
                             model: torch.nn.Module, 
                             x: torch.tensor, 
                             y: torch.tensor, 
                             optimizer: torch.optim.Optimizer) -> float:
        model.train()
    
        x_adv = self.pgd_attack(model, x, y)
    
        logits_adv = model(x_adv)
    
        loss = F.cross_entropy(logits_adv, y)
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        return loss.item()

    def defensive_distillation_loss(self,
                    model: torch.nn.Module, 
                    x: torch.tensor, 
                    y: torch.tensor, # not used
                    optimizer: torch.optim.Optimizer) -> Tuple[float, torch.nn.Module]:

        model.train()

        with torch.no_grad():
            teacher_logits = self.teacher_model(x)
            soft_targets = F.softmax(teacher_logits/self.temperature, dim=1)

        student_logits = model(x)

        loss = F.kl_div(
            F.log_softmax(student_logits/self.temperature, dim=1),
            soft_targets,
            reduction="batchmean"
        ) * (self.temperature * self.temperature)
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        return loss.item(), model

    def trades_loss(self,
                    model: torch.nn.Module, 
                    x: torch.tensor, 
                    y: torch.tensor, 
                    optimizer: torch.optim.Optimizer) -> Tuple[float, torch.nn.Module]:

        # loss = CE(f(x), y) + beta * KullbackLeibler(f(x_adv) || f(x))
        model.train()
    
        x_adv = x.detach() + 0.001 * torch.randn_like(x)
    
        for _ in range(iters):
            x_adv.requires_grad_()
    
            loss_kl = F.kl_div(
                F.log_softmax(model(x_adv), dim=1),
                F.softmax(model(x), dim=1),
                reduction='batchmean'
            )
    
            grad = torch.autograd.grad(loss_kl, x_adv)[0]
    
            x_adv = x_adv.detach() + self.alpha * torch.sign(grad)
            x_adv = torch.min(torch.max(x_adv, x - self.eps), x + self.eps)
            x_adv = torch.clamp(x_adv, 0, 1)
    
        logits = model(x)
        logits_adv = model(x_adv)
    
        loss_natural = F.cross_entropy(logits, y)
    
        loss_robust = F.kl_div(
            F.log_softmax(logits_adv, dim=1),
            F.softmax(logits, dim=1),
            reduction='batchmean'
        )
    
        loss = loss_natural + self.beta * loss_robust
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        return loss.item(), model

    def mart_loss(self,
                  model: torch.nn.Module, 
                  x: torch.tensor, 
                  y: torch.tensor, 
                  optimizer: torch.optim.Optimizer) -> Tuple[float, torch.nn.Module]:

        # x_adv comes from PDF attack
        # loss focusses on misclassified examples and on margin-aware loss for correctly classified examples
        # Misclassification Aware adveRsarial Training (MART)
        model.train()
    
        x_adv = self.pgd_attack(model, x, y)
    
        logits = model(x)
        logits_adv = model(x_adv)
    
        prob = F.softmax(logits, dim=1)
        prob_adv = F.softmax(logits_adv, dim=1)
    
        ce_loss = F.cross_entropy(logits_adv, y)
    
        tmp1 = torch.argsort(prob_adv, dim=1)[:, -2:]
        new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])
    
        loss_adv = ce_loss + F.nll_loss(
            torch.log(1.0001 - prob_adv + 1e-12),
            new_y
        )
    
        true_prob = prob.gather(1, y.unsqueeze(1)).squeeze()
    
        kl = F.kl_div(
            torch.log(prob_adv + 1e-12),
            prob,
            reduction='none'
        ).sum(dim=1)
    
        loss_robust = (kl * (1 - true_prob)).mean()
    
        loss = loss_adv + self.beta * loss_robust
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        return loss.item(), model

    def bayesian_loss(self,
                      model: torch.nn.Module, 
                      x: torch.tensor, 
                      y: torch.tensor, 
                      optimizer: torch.optim.Optimizer) -> Tuple[float, torch.nn.Module]:

        model.train()
    
        logits = model(x)
        loss = F.cross_entropy(logits, y)
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        return loss.item(), model    

    def get_defense(self, 
                    defense_type: DefenseType,
                    data_loader: torch.utils.data.DataLoader,
                    epochs: int,
                   ) -> Callable[..., Any]:
        if defense_type == DefenseType.DISTILLATION:
            self.train_teacher_model(data_loader, epochs)
            return self.defensive_distillation_loss
        elif defense_type == DefenseType.TRADES:
            return self.trades_loss
        elif defense_type == DefenseType.MART:
            return self.mart_loss
        elif defense_type == DefenseType.BAYESIAN:
            return self.bayesian_loss   
        raise ValueError(f"Unsupported defense type {defense_type}")

    def train_teacher_model(self, 
                            data_loader: torch.utils.data.DataLoader,
                            epochs: int,
                           ):

    self.teacher_model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    for _ in range(epochs):
        for x,y in data_loader:
            logits = self.teacher_model(x)
    
            loss = F.cross_entropy(logits/self.temperature, y)
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    self.teacher_model.eval()

    def apply_defense(self,
                      model: torch.nn.Module, 
                      optimizer: torch.optim.Optimizer,
                      data_loader: torch.utils.data.DataLoader,
                      test_data_loader: torch.utils.data.DataLoader,
                      epochs: int,
                      defense_type: DefenseType
                     ) -> None:
        defense = self.get_defense(defense_type, data_loader, epochs)
        
        for epoch in range(epochs):
            epoch_loss = 0
            for x, y in data_loader:
                x, y = x.to(device), y.to(device)
                loss, model = defense(model, x, y, optimizer)
                epoch_loss += loss
            print(f"Epoch {epoch} | Loss {epoch_loss:.4f}")

        self.evaluate_defense(model, test_data_loader)

    def evaluate_defense(self,
                         model: torch.nn.Module, 
                         data_loader: torch.utils.data.DataLoader,
                         device
                        ) -> None:

        attack = AdversarialAttack()
        count, max_batch = 0, 0
        for x, labels in data_loader:
            print(f"## Batch {count}/{max_batch} ##")
            x = x.to(device)
            labels = labels.to(device)
            attack.fgsm_attack(model, x, labels)
            attack.pgd_attack(model, x, labels)
            attack.cw_attack(model, x, labels)
            
            attack.run_attacks(model, x, labels, 'Linf')
            attack.run_attacks(model, x, labels, 'L2')
            count += 1
            break

