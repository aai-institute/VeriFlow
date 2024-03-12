from typing import Iterable
from src.veriflow.transforms import Rotation, CompositeRotation
from src.veriflow.linalg import random_orthonormal_matrix
import torch
from torch.distributions import constraints
import pyro
from pyro.distributions.torch_distribution import TorchDistributionMixin
import math

class RotatedLaplace(torch.distributions.Distribution):
    """Implements a Laplace distribution that is rotated so that the bounding box of the density contours is of minimal (Euclidean) volume."""
    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    support = constraints.real
    has_enumerate_support = False  
    
    def __init__(self, loc: torch.Tensor, scale: torch.Tensor):
        self.dim = loc.shape[0]
        self.loc = loc
        self.scale = scale
        self.rotation = random_orthonormal_matrix(self.dim)
        self.laplace = pyro.distributions.Laplace(self.loc, self.scale)
        batch_shape = self.laplace.batch_shape
        if len(batch_shape) > 0:
            self.laplace = pyro.distributions.Independent(
                self.laplace, len(batch_shape)
            )
        self.shape = self.laplace.event_shape
        super().__init__(event_shape=(self.dim,), validate_args=False)
    
    def sample(self, sample_shape: Iterable[int] = None) -> torch.Tensor:
        """Samples n points from the distribution."""
        if sample_shape is None:
            sample_shape = ()
        else:
            sample_shape = tuple(sample_shape)
        
        return torch.matmul(self.laplace.sample(sample_shape), self.rotation)
    
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the log probability of the points x under the distribution."""
        return self.laplace.log_prob(torch.matmul(x, self.rotation.t()))
    
class UniformUnitLpBall(torch.distributions.Distribution):
    """Implements a uniform distribution on the unit ball."""
    
    support = constraints.real
    has_enumerate_support = False  
    
    def __init__(self, dim: int, p: float):
        self.p = p
        self.dim = dim
        if self.p == 1: 
            self.log_surface_area_unit_ball = (3/2) * math.log(self.dim) + math.log(2) * self.dim  -  torch.log(torch.arange(1, self.dim + 1)).sum() 
        elif self.p == 2:
            self.log_surface_area_unit_ball = math.log(2) + (self.dim / 2) * math.log(math.pi) - torch.lgamma(self.dim / 2)
        elif self.p == math.inf:
            self.log_surface_area_unit_ball = math.log(2) * self.dim + math.log(self.dim) 
        else:
            raise ValueError("p must be 1, 2, or inf.")
        super().__init__(event_shape=(dim,), validate_args=False)
        
    def sample(self, sample_shape: Iterable[int] = None) -> torch.Tensor:
        """Samples batch of shape sample_shape from the distribution."""
        if sample_shape is None:
            sample_shape = ()
        else:
            sample_shape = tuple(sample_shape)
            
        if self.p == 1:
            x = pyro.distributions.Dirichlet(torch.ones(self.dim)).sample(sample_shape)
            dims = pyro.distributions.Categorical(probs=torch.ones(self.dim, 2) / 2).sample(sample_shape) * 2 - 1
            x = x * dims
        elif self.p == 2:
            x = pyro.distributions.Normal(0, 1).sample(sample_shape + (self.dim,))
            x = x / x.norm(dim=-1, keepdim=True)
        elif self.p == math.inf:
            extremal_dims = pyro.distributions.Categorical(torch.ones(self.dim)/ self.dim).sample(sample_shape + (1,))
            mask = torch.ones(sample_shape + (self.dim,)).cumsum(dim=-1) - 1 == extremal_dims

            boundary = torch.ones(sample_shape + (self.dim,))
            hyperplane_distribution = pyro.distributions.Uniform(-boundary, boundary)
            x = hyperplane_distribution.sample()
            x[mask] = 1.
        else:
            raise ValueError("p must be 1, 2, or inf.")
        
        return x
    
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the log probability of the points x under the distribution."""
        
        return -self.log_surface_area_unit_ball
    
    
class RadialDistribution(torch.distributions.Distribution):
    """Implements radial distributions. More precisely, this class realizes Lp-radial distributions with specifiable redial distribution.
    
    Args:
        loc: Location of the distribution
        radial_distribution: Distribution of the radial component
        p: Exponent of the Lp norm used to define the distribution. Currently p = 1, 2, and inf are supported.
    """
    arg_constraints = {"loc": constraints.real}
    support = constraints.real
    has_enumerate_support = False  
    
    def __init__(self, loc: torch.Tensor, radial_distribution: torch.distributions.Distribution, p: float):
        if not isinstance(p, float):
            raise ValueError("p must be a float.")
        if p <= 0:
            raise ValueError("p must be positive.")
        
        self.loc = loc
        self.radial_distribution = radial_distribution
        self.p = p
        self.dim = loc.shape[0]
        self.unit_ball_distribution = UniformUnitLpBall(self.dim, p)
        
        super().__init__(event_shape=(loc.shape[0],), validate_args=False)
        
    def sample(self, sample_shape: Iterable[int] = None) -> torch.Tensor:
        """Samples batch of shape sample_shape from the distribution."""
        if sample_shape is None:
            sample_shape = ()
        else:
            sample_shape = tuple(sample_shape)
        
        r = self.radial_distribution.sample(sample_shape)
        u = self.unit_ball_distribution.sample(sample_shape)
        x = r * u
 
        return x + self.loc
    
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the log probability of the points x under the distribution."""
        x = x - self.loc
        r = x.norm(dim=-1, p=self.p)
        log_prob_radial = self.radial_distribution.log_prob(r)
        log_prob_unit_ball = self.unit_ball_distribution.log_prob(x / r.unsqueeze(-1))
        
        conv_log_abs_det_jacobian = (self.dim - 1) * torch.log(r)
        
        return log_prob_radial + log_prob_unit_ball - conv_log_abs_det_jacobian 