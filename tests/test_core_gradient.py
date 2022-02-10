import os
import unittest
import jax
import jax.numpy as jnp
from hydra import compose, initialize

from leniax.utils import get_container
from leniax import core as leniax_core
from leniax import helpers as leniax_helpers
from leniax import kernels as leniax_kernels

cfd = os.path.dirname(os.path.realpath(__file__))
fixture_dir = os.path.join(cfd, 'fixtures')


class TestCore(unittest.TestCase):
    def test_update_fn_grad(self):
        with initialize(config_path='fixtures'):
            omegaConf = compose(config_name="orbium-test")
            config = get_container(omegaConf, fixture_dir)

        rng_key = jax.random.PRNGKey(0)

        cells, K, mapping = leniax_helpers.init(config)
        gf_params = mapping.get_gf_params()
        kernels_weight_per_channel = mapping.get_kernels_weight_per_channel()
        target = jnp.ones_like(cells) * 0.5

        world_params = config['world_params']
        dt = 1. / jnp.array(world_params['T'])
        get_state_fn_slug = world_params['get_state_fn_slug'] if 'get_state_fn_slug' in world_params else 'v1'
        update_fn = leniax_helpers.build_update_fn(K.shape, mapping, get_state_fn_slug)

        def apply_update(rng_key, cells, K, gf_params, kernels_weight_per_channel, dt, target):
            rng_key, cells_out, _, _ = update_fn(rng_key, cells, K, gf_params, kernels_weight_per_channel, dt)
            error = jnp.sum((cells_out - target)**2)

            return error

        update_fn_cellsgrad = jax.grad(apply_update, argnums=1)
        update_fn_Kgrad = jax.grad(apply_update, argnums=2)
        update_fn_gfngrad = jax.grad(apply_update, argnums=3)
        update_fn_kernelsweightgrad = jax.grad(apply_update, argnums=4)

        # out = update_fn(cells, K, gf_params, kernels_weight_per_channel, dt)
        out_grad = update_fn_cellsgrad(rng_key, cells, K, gf_params, kernels_weight_per_channel, dt, target)
        assert float(jnp.sum(out_grad)) != 0.

        out_grad = update_fn_Kgrad(rng_key, cells, K, gf_params, kernels_weight_per_channel, dt, target)
        assert float(jnp.real(jnp.sum(out_grad))) != 0.

        out_grad = update_fn_gfngrad(rng_key, cells, K, gf_params, kernels_weight_per_channel, dt, target)
        assert float(jnp.sum(out_grad)) != 0.

        out_grad = update_fn_kernelsweightgrad(rng_key, cells, K, gf_params, kernels_weight_per_channel, dt, target)
        assert float(jnp.sum(out_grad)) != 0.

    def test_get_field_grad(self):
        nb_channels = 2
        nb_kernels = 3
        mapping = leniax_kernels.KernelMapping(nb_channels, nb_kernels)
        mapping.kernels_weight_per_channel = [[.5, .4, .0], [0., 0., .2]]
        mapping.cin_gfs = [['poly_quad4', 'poly_quad4'], ['poly_quad4']]
        mapping.cin_gf_params = [[[0.1, 0.1], [0.2, 0.2]], [[0.3, 0.3]]]
        average_weight = True

        get_field = leniax_helpers.build_get_field_fn(mapping.cin_gfs, average_weight)

        def apply_update(potential, gf_params, kernels_weight_per_channel, target):
            out = get_field(potential, gf_params, kernels_weight_per_channel)
            error = jnp.sum((out - target)**2)

            return error

        get_field_potentialgrad = jax.grad(apply_update, argnums=0)
        get_field_gfnparamsgrad = jax.grad(apply_update, argnums=1)
        get_field_kernelsweightgrad = jax.grad(apply_update, argnums=2)

        potential = jnp.array([[[
            [0.3, 0.3],
            [0.3, 0.3],
        ], [
            [0.6, 0.6],
            [0.6, 0.6],
        ], [
            [0.8, 0.8],
            [0.8, 0.8],
        ]]])
        gf_params = mapping.get_gf_params()
        kernels_weight_per_channel = mapping.get_kernels_weight_per_channel()
        target = jnp.ones([1, nb_channels, 2, 2]) * .5

        # out = get_field(potential, gf_params, kernels_weight_per_channel)
        out_grad = get_field_potentialgrad(potential, gf_params, kernels_weight_per_channel, target)
        assert float(jnp.sum(out_grad)) != 0.

        out_grad = get_field_gfnparamsgrad(potential, gf_params, kernels_weight_per_channel, target)
        assert float(jnp.sum(out_grad)) != 0.

        out_grad = get_field_kernelsweightgrad(potential, gf_params, kernels_weight_per_channel, target)
        assert float(jnp.sum(out_grad)) != 0.

    def test_weighted_mean_grad(self):
        def apply_update(field, weights, target):
            out = leniax_core.weighted_mean(field, weights)
            error = jnp.sum((out - target)**2)

            return error

        weighted_mean_fieldgrad = jax.grad(apply_update, argnums=0)
        weighted_mean_weightsgrad = jax.grad(apply_update, argnums=1)

        nb_channels = 2
        field = jnp.array([[[
            [0.3, 0.3],
            [0.3, 0.3],
        ], [
            [0.6, 0.6],
            [0.6, 0.6],
        ], [
            [0.8, 0.8],
            [0.8, 0.8],
        ]]])
        weights = jnp.array([[.5, .4, .0], [.5, .2, .0]])
        target = jnp.ones([1, nb_channels, 2, 2]) * .5

        # out = leniax_core.weighted_mean(field, weights)
        out_grad = weighted_mean_fieldgrad(field, weights, target)
        assert float(jnp.sum(out_grad)) != 0.

        out_grad = weighted_mean_weightsgrad(field, weights, target)
        assert float(jnp.sum(out_grad)) != 0.

    def test_get_state_v1_grad(self):
        rng_key = jax.random.PRNGKey(0)

        def apply_update(rng_key, cells, field, dt, target):
            rng_key, cells = leniax_core.get_state(rng_key, cells, field, dt)
            rng_key, out = leniax_core.get_state(rng_key, cells, field, dt)
            error = jnp.sum((out - target)**2)

            return error

        get_state_grad = jax.grad(apply_update, argnums=1)
        get_state_fieldgrad = jax.grad(apply_update, argnums=2)

        world_shape = [2, 2]
        cells = jnp.ones(world_shape) * .5
        cells = cells.at[0, 0].set(1.)
        field = jnp.ones(world_shape) * .2
        dt = jnp.array(1. / 3.)
        target = jnp.ones(world_shape) * 0.7

        # out = leniax_core.get_state(cells, field, dt)
        out_grad = get_state_grad(rng_key, cells, field, dt, target)
        assert float(jnp.sum(out_grad)) != 0.

        out_grad = get_state_fieldgrad(rng_key, cells, field, dt, target)
        assert float(jnp.sum(out_grad)) != 0.

    def test_get_state_v2_grad(self):

        rng_key = jax.random.PRNGKey(0)

        def apply_update(rng_key, cells, field, dt, target):
            rng_key, cells = leniax_core.get_state_v2(rng_key, cells, field, dt)
            rng_key, out = leniax_core.get_state_v2(rng_key, cells, field, dt)
            error = jnp.sum((out - target)**2)

            return error

        get_state_cellsgrad = jax.grad(apply_update, argnums=1)
        get_state_fieldgrad = jax.grad(apply_update, argnums=2)

        world_shape = [2, 2]
        cells = jnp.ones(world_shape) * .5
        field = jnp.ones(world_shape) * 3
        dt = jnp.array(1. / 3.)
        target = jnp.ones(world_shape) * 0.7

        # out = leniax_core.get_state_v2(cells, field, dt)
        out_grad = get_state_cellsgrad(rng_key, cells, field, dt, target)
        assert float(jnp.sum(out_grad)) != 0.

        out_grad = get_state_fieldgrad(rng_key, cells, field, dt, target)
        assert float(jnp.sum(out_grad)) != 0.
