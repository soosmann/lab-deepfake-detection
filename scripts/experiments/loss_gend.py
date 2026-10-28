import torch
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# Losses para LNCLIP-DF / GenD
#
# 1. BCE                — clasificación binaria estándar
# 2. Uniformity loss    — distribuye features uniformemente en la hiperesfera
# 3. Alignment loss     — acerca features de la misma clase
# 4. Slerp              — augmentación en espacio latente (hiperesfera)
#
# Referencia:
#   Wang & Isola, "Understanding Contrastive Representation Learning through
#   Alignment and Uniformity on the Hypersphere", ICML 2020.
# ─────────────────────────────────────────────────────────────────────────────

def bce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Binary cross-entropy estándar.
    logits : [B, 1] o [B]
    labels : [B]  con valores 0.0 o 1.0
    """
    return F.binary_cross_entropy_with_logits(
        logits.squeeze(-1), labels.float()
    )


def uniformity_loss(features: torch.Tensor, t: float = 2.0) -> torch.Tensor:
    """
    Uniformity loss (Wang & Isola 2020).
    Distribuye los features uniformemente sobre la hiperesfera unitaria.
    Features deben estar L2-normalizados.

    L_unif = log( mean( exp(-t * ||zi - zj||²) ) )

    Args:
        features : [B, D]  — L2-normalizados
        t        : temperatura (paper usa 2.0)
    """
    # Distancia cuadrada entre todos los pares: ||zi - zj||² = 2 - 2*zi·zj
    sq_pdist = 2.0 - 2.0 * (features @ features.T)  # [B, B]
    sq_pdist = sq_pdist.clamp(min=0)

    # Quitar diagonal (zi == zi → dist=0)
    B = features.size(0)
    mask = ~torch.eye(B, dtype=torch.bool, device=features.device)
    sq_pdist = sq_pdist[mask]  # [B*(B-1)]

    return torch.log(torch.exp(-t * sq_pdist).mean() + 1e-8)


def alignment_loss(features: torch.Tensor, labels: torch.Tensor,
                   alpha: float = 2.0) -> torch.Tensor:
    """
    Alignment loss (Wang & Isola 2020).
    Acerca los features de la misma clase.
    Features deben estar L2-normalizados.

    L_align = mean( ||zi - zj||^alpha ) para pares (i,j) de la misma clase

    Args:
        features : [B, D]  — L2-normalizados
        labels   : [B]     — 0.0 o 1.0
        alpha    : exponente (paper usa 2.0)
    """
    B = features.size(0)
    labels = labels.float()

    # Máscara de pares de la misma clase
    same_class = (labels.unsqueeze(0) == labels.unsqueeze(1))  # [B, B]
    same_class = same_class & ~torch.eye(B, dtype=torch.bool,
                                         device=features.device)

    if same_class.sum() == 0:
        return torch.tensor(0.0, device=features.device)

    # Distancia euclidea al cuadrado
    sq_pdist = 2.0 - 2.0 * (features @ features.T)
    sq_pdist = sq_pdist.clamp(min=0)

    if alpha == 2.0:
        dist = sq_pdist[same_class]
    else:
        dist = (sq_pdist[same_class] + 1e-8).pow(alpha / 2.0)

    return dist.mean()


def slerp(z1: torch.Tensor, z2: torch.Tensor,
          t: float) -> torch.Tensor:
    """
    Spherical Linear Interpolation entre dos features L2-normalizados.
    slerp(z1, z2, t) = sin((1-t)*θ)/sin(θ) * z1 + sin(t*θ)/sin(θ) * z2
    donde θ = arccos(z1 · z2)

    Para t ∈ (0, 1) produce un punto en el arco geodésico entre z1 y z2.

    Args:
        z1, z2 : [B, D]  — L2-normalizados
        t      : float ∈ [0, 1]
    Returns:
        z_slerp : [B, D]  — también L2-normalizados (en la hiperesfera)
    """
    # cos(θ) = z1 · z2, clamp para evitar acos fuera de [-1, 1]
    cos_theta = (z1 * z2).sum(dim=-1, keepdim=True).clamp(-1 + 1e-7, 1 - 1e-7)
    theta = torch.acos(cos_theta)  # [B, 1]

    sin_theta = torch.sin(theta).clamp(min=1e-7)

    coeff1 = torch.sin((1.0 - t) * theta) / sin_theta  # [B, 1]
    coeff2 = torch.sin(t * theta) / sin_theta           # [B, 1]

    return coeff1 * z1 + coeff2 * z2


def slerp_augmentation(features: torch.Tensor,
                       labels: torch.Tensor,
                       t_range: tuple = (0.3, 0.7)):
    """
    Para cada feature xi, muestreamos aleatoriamente un xj de la MISMA CLASE
    en el mismo batch y computamos slerp(xi, xj, t) con t aleatorio.

    Duplica el batch añadiendo las interpolaciones.
    Esto enriquece el espacio latente y hace la clase fake más robusta
    a métodos de generación no vistos.

    Args:
        features : [B, D]  — L2-normalizados
        labels   : [B]
        t_range  : rango de t para la interpolación
    Returns:
        (aug_features, aug_labels) : [2B, D] y [2B]
    """
    B = features.size(0)
    labels_f = labels.float()

    aug_features = []
    aug_labels   = []

    for i in range(B):
        # Encontrar índices de la misma clase
        same = (labels_f == labels_f[i]).nonzero(as_tuple=True)[0]
        same = same[same != i]  # excluir el propio índice

        if len(same) == 0:
            # No hay par disponible → usar el mismo feature sin augmentar
            aug_features.append(features[i])
        else:
            j = same[torch.randint(len(same), (1,)).item()].item()
            t = torch.empty(1).uniform_(*t_range).item()
            z_interp = slerp(features[i].unsqueeze(0),
                             features[j].unsqueeze(0), t)
            aug_features.append(z_interp.squeeze(0))

        aug_labels.append(labels_f[i])

    aug_features = torch.stack(aug_features)
    aug_labels   = torch.stack(aug_labels)

    # Concatenar original + augmentado
    all_features = torch.cat([features, aug_features], dim=0)
    all_labels   = torch.cat([labels_f, aug_labels],   dim=0)

    return all_features, all_labels


class GenDLoss(torch.nn.Module):
    """
    Loss total = BCE + λ_unif * Uniformity + λ_align * Alignment

    Pesos por defecto del paper GenD.
    """

    def __init__(self, lambda_unif: float = 0.5,
                 lambda_align: float = 1.0,
                 t_unif: float = 2.0,
                 alpha_align: float = 2.0):
        super().__init__()
        self.lambda_unif  = lambda_unif
        self.lambda_align = lambda_align
        self.t_unif       = t_unif
        self.alpha_align  = alpha_align

    def forward(self, logits: torch.Tensor,
                features: torch.Tensor,
                labels: torch.Tensor):
        """
        Args:
            logits   : [B, 1]  — salida del clasificador
            features : [B, D]  — CLS token L2-normalizados
            labels   : [B]     — 0.0 (fake) o 1.0 (real)
        Returns:
            loss_total, dict con componentes
        """
        l_bce   = bce_loss(logits, labels)
        l_unif  = uniformity_loss(features, t=self.t_unif)
        l_align = alignment_loss(features, labels, alpha=self.alpha_align)

        total = l_bce + self.lambda_unif * l_unif + self.lambda_align * l_align

        return total, {
            "bce":   l_bce.item(),
            "unif":  l_unif.item(),
            "align": l_align.item(),
            "total": total.item(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    B, D = 8, 1024
    features = F.normalize(torch.randn(B, D), dim=-1)
    labels   = torch.tensor([1., 1., 1., 1., 0., 0., 0., 0.])
    logits   = torch.randn(B, 1)

    criterion = GenDLoss()
    loss, components = criterion(logits, features, labels)

    print(f"Loss total : {loss.item():.4f}")
    print(f"Components : {components}")

    # Test slerp
    aug_f, aug_l = slerp_augmentation(features, labels)
    print(f"Slerp aug features : {aug_f.shape}")  # [16, 1024]
    print(f"Slerp aug norms    : {aug_f.norm(dim=-1).mean():.4f}")  # ~1.0
    print("✅ loss_gend.py OK")