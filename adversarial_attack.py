import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchattacks import AutoAttack, Square
import foolbox as fb
from foolbox.attacks import BoundaryAttack

class AdversarialAttack:
    def __init__(self,
                 n_queries: int = 5000, 
                 eps: float = 8/255
                ) -> None:
        self.n_queries = n_queries
        self.eps = eps
        
    def square_attack(self, 
                      model: torch.nn.Module, 
                      x: torch.tensor, 
                      labels: torch.tensor, 
                      norm='Linf', 
                     ) -> None:
        model.eval()
        
        attack = Square(model, norm=norm, eps=self.eps, n_queries=self.n_queries, n_restarts=1)
        
        # Generate adversarial examples
        x_adv = attack(x, labels)
        
        # Evaluate the adversarial examples
        self.evaluate(x, x_adv, labels)

    def fab_attack(self, 
                   model: torch.nn.Module, 
                   x: torch.tensor, 
                   labels: torch.tensor, 
                   norm='Linf', 
                  ) -> None:
        model.eval()
        
        adversary = AutoAttack(
            model,
            norm=norm,
            eps=self.eps,
            version='custom'
        )
        
        # Use only FAB attack
        adversary.attacks_to_run = ['fab']
        
        # Generate adversarial examples
        x_adv = adversary.run_standard_evaluation(
            x,
            labels,
            bs=32
        )
        
        # Evaluate the adversarial examples
        self.evaluate(x, x_adv, labels)

    def boundary_attack(self, 
                        model: torch.nn.Module, 
                        x: torch.tensor, 
                        labels: torch.tensor, 
                        norm='Linf', 
                       ) -> None:
        model.eval()
        # Wrap model for Foolbox
        fmodel = fb.PyTorchModel(model, bounds=(0,1))
        # Boundary attack
        attack = BoundaryAttack()
        
        raw_adv, clipped_adv, success = attack(
            fmodel,
            x,
            labels,
            epsilons=None,
        )
        # Evaluate the adversarial examples
        self.evaluate(x, clipped_adv, labels)
    
    def pgd_attack(self, 
                   model: torch.nn.Module, 
                   x: torch.tensor, 
                   labels: torch.tensor, 
                   norm='Linf', 
                  ) -> None:
        model.eval()
        adversary = AutoAttack(model, norm=norm, eps=self.eps, attacks_to_run=['apgd-ce', 'apgd-dlr'])
        x_adv = adversary.run_attack(x, labels)
        self.evaluate(x, x_adv, labels)
        
    
    def attack_evaluation(self,
                        model: torch.nn.Module, 
                        x: torch.tensor, 
                        labels: torch.tensor, 
                        norm='Linf',
                       ) -> None:
        adversary = AutoAttack(model, norm=norm, eps=self.eps, version='custom', attacks_to_run=['apgd-ce', 'apgd-dlr'])
        
        # Setting n_restarts lower makes it faster, but less thorough
        adversary.apgd.n_restarts = 1 
        adversary.apgd.n_iter = self.n_queries # Number of iterations for boundary optimization
        # Run the attack
        x_adv = adversary.run_standard_evaluation(x, labels)
        self.evaluate(x, x_adv, labels)

    def attack_success_rate(self,
                           preds: torch.tensor,
                           preds_adv: torch.tensor,
                           labels: torch.tensor,
                           ) -> float:
        correct = (preds == labels)
        misclassified = (preds_adv[correct] != labels[correct]).sum().item()
        return misclassified / correct.sum().item()

    def perturbation_size(self,
                          x: torch.tensor,
                          x_adv: torch.tensor,
                          norm: int | None = 2
                         ) -> float:
        perturbation = x_adv - x

        if norm is not None:
            val = torch.norm(
                perturbation.view(perturbation.size(0), -1),
                p=2,
                dim=1
            ).mean().item()
        else:
            val = perturbation.abs().view(perturbation.size(0), -1).max(dim=1)[0].mean().item()
        return val

    def confidence_drop(self,
                        orig_output: torch.tensor,
                        adv_output: torch.tensor,
                       ) -> float:
        orig_probs = F.softmax(orig_output, dim=1)
        adv_probs = F.softmax(adv_output, dim=1)
        drop = (
            orig_probs.max(dim=1)[0] - adv_probs.max(dim=1)[0]
        ).mean().item()
        return drop


    def evaluate(self,
                 x: torch.tensor,
                 x_adv: torch.tensor,
                 labels: torch.tensor, 
                ) -> None:
        # -----------------------------
        # Evaluation
        # -----------------------------
        
        with torch.no_grad():
            # ---- Clean prediction ----
            outputs = model(x)
            preds = outputs.argmax(1)
            clean_correct = (preds == labels).sum().item()

            # ---- Adversarial prediction ----
            outputs_adv = model(x_adv)
            preds_adv = outputs_adv.argmax(1)
            adv_correct = (preds_adv == labels).sum().item()

            # attack success rate
            asr = self.attack_success_rate(preds, preds_adv, labels)

            # perturbation size
            l2_sz = self.perturbation_size(x, x_adv, norm=2)
            linf_size = self.perturbation_size(x, x_adv)

            # confidence drop
            conf_drop = self.confidence_drop(outputs, outputs_adv)
            
    
        total = images.size(0)
    
        # -----------------------------
        # Results
        # -----------------------------
        clean_acc = 100 * clean_correct / total
        adv_acc = 100 * adv_correct / total
        
        print("\n===== RESULTS =====")
        print(f"Images evaluated: {total}")
        print(f"Clean Accuracy: {clean_acc:.4f}%")
        print(f"Robust Accuracy: {adv_acc:.4f}%") # robust accuracy = 1 - ASR
        print(f"ASR: {asr:.2f}%")
        print(f"Mean L2 Perturbation Size: {l2_sz:.4f}%")
        print(f"Mean Linf Perturbation Size: {linf_sz:.4f}%")
        print(f"Confidence Drop: {conf_drop:.4f}%")
        print("===================")
