"""Loss functions for FSRA project."""

from .triplet_loss import TripletLoss, HardTripletLoss
from .combined_loss import CombinedLoss

__all__ = ['TripletLoss', 'HardTripletLoss', 'CombinedLoss']
