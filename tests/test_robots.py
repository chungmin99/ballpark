"""Tests for robot spherization.

Tests panda, ur5, and yumi robots from robot_descriptions.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import robot_descriptions
    ROBOT_DESCRIPTIONS_AVAILABLE = True
except ImportError:
    ROBOT_DESCRIPTIONS_AVAILABLE = False

from ballpark import Robot, spherize
from ballpark._spherize import compute_coverage

from .conftest import BUDGETS, Timer


# =============================================================================
# ROBOT CONFIGURATIONS
# =============================================================================

ROBOTS = [
    {
        "name": "panda",
        "module": "robot_descriptions.panda_description",
        "attr": "URDF_PATH",
        "min_links": 5,  # Expected minimum number of links with collision meshes
    },
    {
        "name": "ur5",
        "module": "robot_descriptions.ur5_description",
        "attr": "URDF_PATH",
        "min_links": 5,
    },
    {
        "name": "yumi",
        "module": "robot_descriptions.yumi_description",
        "attr": "URDF_PATH",
        "min_links": 8,  # Dual arm robot
    },
]


def get_robot_urdf_path(robot_config: dict) -> str | None:
    """Get URDF path for a robot configuration."""
    try:
        import importlib
        module = importlib.import_module(robot_config["module"])
        return getattr(module, robot_config["attr"])
    except (ImportError, AttributeError):
        return None


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture(params=[r["name"] for r in ROBOTS])
def robot_config(request):
    """Get robot configuration by name."""
    for r in ROBOTS:
        if r["name"] == request.param:
            return r
    pytest.fail(f"Unknown robot: {request.param}")


# =============================================================================
# SKIP CONDITION
# =============================================================================

requires_robot_descriptions = pytest.mark.skipif(
    not ROBOT_DESCRIPTIONS_AVAILABLE,
    reason="robot_descriptions package not installed"
)


# =============================================================================
# ROBOT LOADING TESTS
# =============================================================================


@requires_robot_descriptions
class TestRobotLoading:
    """Test that robots can be loaded correctly."""

    def test_robot_loads(self, robot_config):
        """Verify robot URDF can be loaded."""
        urdf_path = get_robot_urdf_path(robot_config)
        if urdf_path is None:
            pytest.skip(f"Could not load {robot_config['name']}")

        import yourdfpy
        urdf = yourdfpy.URDF.load(urdf_path, load_collision_meshes=True)

        assert urdf is not None
        assert len(urdf.link_map) > 0

    def test_robot_has_collision_meshes(self, robot_config):
        """Verify robot has collision meshes for links."""
        urdf_path = get_robot_urdf_path(robot_config)
        if urdf_path is None:
            pytest.skip(f"Could not load {robot_config['name']}")

        import yourdfpy
        urdf = yourdfpy.URDF.load(urdf_path, load_collision_meshes=True)

        # Count links with collision geometry
        links_with_collision = 0
        for link_name in urdf.link_map:
            link = urdf.link_map[link_name]
            if link.collisions:
                links_with_collision += 1

        assert links_with_collision >= robot_config["min_links"], (
            f"{robot_config['name']} has only {links_with_collision} links "
            f"with collision meshes, expected at least {robot_config['min_links']}"
        )


# =============================================================================
# ROBOT SPHERIZATION TESTS
# =============================================================================


@requires_robot_descriptions
class TestRobotSpherization:
    """Test robot spherization."""

    @pytest.mark.parametrize("budget", [32, 64, 128])
    def test_robot_spherization_produces_valid_spheres(self, robot_config, budget):
        """Verify robot spherization produces valid spheres for all links."""
        urdf_path = get_robot_urdf_path(robot_config)
        if urdf_path is None:
            pytest.skip(f"Could not load {robot_config['name']}")

        import yourdfpy
        urdf = yourdfpy.URDF.load(urdf_path, load_collision_meshes=True)
        robot = Robot(urdf)

        result = robot.spherize(target_spheres=budget)

        # Should have spheres for at least some links
        assert len(result.link_spheres) > 0, (
            f"{robot_config['name']} produced no link spheres"
        )

        total_spheres = 0
        for link_name, spheres in result.link_spheres.items():
            total_spheres += len(spheres)

            for i, s in enumerate(spheres):
                center = np.asarray(s.center)
                radius = float(s.radius)

                # Check validity
                assert np.all(np.isfinite(center)), (
                    f"{robot_config['name']} link {link_name} sphere {i} "
                    f"has non-finite center"
                )
                assert radius > 0, (
                    f"{robot_config['name']} link {link_name} sphere {i} "
                    f"has non-positive radius"
                )

        print(
            f"\n  {robot_config['name']} @ {budget}: "
            f"{len(result.link_spheres)} links, {total_spheres} total spheres"
        )

    def test_robot_spherization_respects_budget(self, robot_config):
        """Verify total sphere count is reasonable given budget."""
        urdf_path = get_robot_urdf_path(robot_config)
        if urdf_path is None:
            pytest.skip(f"Could not load {robot_config['name']}")

        import yourdfpy
        urdf = yourdfpy.URDF.load(urdf_path, load_collision_meshes=True)
        robot = Robot(urdf)

        budget = 64
        result = robot.spherize(target_spheres=budget)

        total_spheres = sum(len(s) for s in result.link_spheres.values())

        # Total should not exceed 2x budget
        assert total_spheres <= budget * 2, (
            f"{robot_config['name']} produced {total_spheres} spheres, "
            f"exceeding 2x budget of {budget}"
        )


# =============================================================================
# PER-LINK TESTS
# =============================================================================


@requires_robot_descriptions
class TestPerLinkSpherization:
    """Test spherization of individual robot links."""

    def test_individual_link_coverage(self, robot_config):
        """Test coverage for individual links."""
        urdf_path = get_robot_urdf_path(robot_config)
        if urdf_path is None:
            pytest.skip(f"Could not load {robot_config['name']}")

        import yourdfpy
        urdf = yourdfpy.URDF.load(urdf_path, load_collision_meshes=True)
        robot = Robot(urdf)

        result = robot.spherize(target_spheres=64)

        # Test coverage for a few links
        links_tested = 0
        for link_name, spheres in result.link_spheres.items():
            if len(spheres) == 0:
                continue

            # Get link mesh from robot's cached meshes
            mesh = robot._link_meshes.get(link_name)
            if mesh is None or mesh.is_empty:
                continue

            n_samples = 1000
            points = np.asarray(mesh.sample(n_samples))
            coverage = compute_coverage(points, spheres)

            # Links should achieve reasonable coverage
            assert coverage >= 0.5, (
                f"{robot_config['name']} link {link_name} "
                f"coverage {coverage:.3f} too low"
            )

            links_tested += 1
            if links_tested >= 3:  # Test first 3 links
                break


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================


@requires_robot_descriptions
@pytest.mark.slow
class TestRobotPerformance:
    """Performance tests for robot spherization."""

    def test_spherization_timing(self, robot_config):
        """Track spherization timing for robots."""
        urdf_path = get_robot_urdf_path(robot_config)
        if urdf_path is None:
            pytest.skip(f"Could not load {robot_config['name']}")

        import yourdfpy
        urdf = yourdfpy.URDF.load(urdf_path, load_collision_meshes=True)
        robot = Robot(urdf)

        budgets = [32, 64, 128]
        for budget in budgets:
            timer = Timer()
            with timer:
                result = robot.spherize(target_spheres=budget)

            total_spheres = sum(len(s) for s in result.link_spheres.values())
            print(
                f"\n  {robot_config['name']} @ {budget}: "
                f"{total_spheres} spheres in {timer.elapsed_ms:.1f}ms"
            )


# =============================================================================
# JSON EXPORT TESTS
# =============================================================================


@requires_robot_descriptions
class TestRobotExport:
    """Test robot spherization export functionality."""

    def test_json_export(self, robot_config, tmp_path):
        """Test JSON export of robot spheres."""
        urdf_path = get_robot_urdf_path(robot_config)
        if urdf_path is None:
            pytest.skip(f"Could not load {robot_config['name']}")

        import json
        import yourdfpy
        urdf = yourdfpy.URDF.load(urdf_path, load_collision_meshes=True)
        robot = Robot(urdf)

        result = robot.spherize(target_spheres=32)

        # Export to JSON
        output_path = tmp_path / f"{robot_config['name']}_spheres.json"
        result.save_json(output_path)

        # Verify file exists and is valid JSON
        assert output_path.exists()

        with open(output_path) as f:
            data = json.load(f)

        # JSON format: {link_name: {centers: [...], radii: [...]}, ...}
        assert len(data) == len(result.link_spheres)
        for link_name in result.link_spheres:
            if link_name in data:
                assert "centers" in data[link_name]
                assert "radii" in data[link_name]
