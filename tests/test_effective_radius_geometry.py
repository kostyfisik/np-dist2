import numpy as np
import pytest
from np_dist2.analysis import calculate_effective_radius

def calculate_cone_volume(radius, half_angle):
    """
    Calculate volume of a cone given the sphere radius and half-angle.
    h_cone = R * cos(theta)
    r_base = R * sin(theta)
    V = (1/3) * pi * r_base^2 * h_cone
    """
    h_cone = radius * np.cos(half_angle)
    r_base = radius * np.sin(half_angle)
    return (1.0/3.0) * np.pi * r_base**2 * h_cone

def calculate_spherical_cap_volume(radius, half_angle):
    """
    Calculate volume of a spherical cap.
    h_cap = R * (1 - cos(theta))
    V = (1/3) * pi * h_cap^2 * (3R - h_cap)
    """
    h_cap = radius * (1.0 - np.cos(half_angle))
    return (1.0/3.0) * np.pi * h_cap**2 * (3 * radius - h_cap)

def calculate_sector_volume_explicit(radius, solid_angle):
    """
    Calculate spherical sector volume by summing cone and cap volumes.
    Omega = 2*pi * (1 - cos(theta))
    """
    # cos(theta) = 1 - Omega / (2*pi)
    cos_theta = 1.0 - solid_angle / (2.0 * np.pi)
    # Clamp for numerical stability
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    half_angle = np.arccos(cos_theta)
    
    v_cone = calculate_cone_volume(radius, half_angle)
    v_cap = calculate_spherical_cap_volume(radius, half_angle)
    return v_cone + v_cap

def calculate_sector_volume_simplified(radius, solid_angle):
    """
    Calculate spherical sector volume using simplified formula.
    V = (Omega / 3) * R^3
    """
    return (solid_angle / 3.0) * radius**3

class TestEffectiveRadiusGeometry:
    
    def test_cone_foundation_match(self):
        """Test that cone base radius matches spherical cap base radius."""
        R = 10.0
        # Test for various solid angles (corresponding to N=10, 100, 1000)
        for N in [10, 100, 1000]:
            omega = 4 * np.pi / N
            cos_theta = 1.0 - omega / (2.0 * np.pi)
            theta = np.arccos(cos_theta)
            
            # Cone geometry
            h_cone = R * np.cos(theta)
            r_cone_base = h_cone * np.tan(theta)
            
            # Cap geometry (sphere slice at h_cone)
            # r_cap_base = sqrt(R^2 - h_cone^2)
            r_cap_base = np.sqrt(R**2 - h_cone**2)
            
            assert np.isclose(r_cone_base, r_cap_base)
            assert np.isclose(r_cone_base, R * np.sin(theta))

    def test_sector_volume_consistency(self):
        """Test that explicit cone+cap volume equals simplified sector volume."""
        R = 5.0
        for N in [6, 12, 100, 1000]:
            omega = 4 * np.pi / N
            
            v_explicit = calculate_sector_volume_explicit(R, omega)
            v_simplified = calculate_sector_volume_simplified(R, omega)
            
            # Check relative error
            assert np.isclose(v_explicit, v_simplified, rtol=1e-10)
            
    def test_total_volume_sphere(self):
        """Test that summing N sectors gives sphere volume."""
        R = 3.0
        expected_vol = (4.0/3.0) * np.pi * R**3
        
        for N in [10, 50, 100]:
            omega = 4 * np.pi / N
            vol_sum = 0.0
            for _ in range(N):
                vol_sum += calculate_sector_volume_explicit(R, omega)
            
            assert np.isclose(vol_sum, expected_vol)

    def test_effective_radius_calculation(self):
        """Test the effective radius calculation with a perfect sphere."""
        # Create atoms on a sphere
        R = 10.0
        # Use Fibonacci sphere for uniform distribution
        num_atoms = 100
        indices = np.arange(0, num_atoms, dtype=float) + 0.5
        phi = np.arccos(1 - 2*indices/num_atoms)
        theta = np.pi * (1 + 5**0.5) * indices
        x, y, z = R * np.cos(theta) * np.sin(phi), R * np.sin(theta) * np.sin(phi), R * np.cos(phi)
        atoms = np.column_stack((x, y, z))
        
        # Calculate effective radius
        # We expect it to be close to R
        r_eff, _ = calculate_effective_radius(atoms, num_directions=1000, cylinder_radius=R/2)
        
        assert np.isclose(r_eff, R, rtol=0.01)

