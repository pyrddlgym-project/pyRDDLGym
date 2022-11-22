import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random
import optax
from typing import Dict, Generator
import warnings

from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLNotImplementedError
from pyRDDLGym.Core.Jax.JaxRDDLCompiler import JaxRDDLCompiler
from pyRDDLGym.Core.Jax.JaxRDDLSimulator import JaxRDDLSimulator
from pyRDDLGym.Core.Parser.rddl import RDDL


class JaxRDDLContinuousCompiler(JaxRDDLCompiler):
    
    def _jax_relational(self, expr, op, params):
        valid_ops = JaxRDDLContinuousCompiler.RELATIONAL_OPS
        JaxRDDLContinuousCompiler._check_valid_op(expr, valid_ops)
        JaxRDDLContinuousCompiler._check_num_args(expr, 2)
        
        warnings.warn('Relational operator {} will be converted to arithmetic.'.format(
            valid_ops.keys()), FutureWarning, stacklevel=2)
                    
        lhs, rhs = expr.args
        jax_lhs = self._jax(lhs, params)
        jax_rhs = self._jax(rhs, params)
        jax_op = valid_ops[op]
        return JaxRDDLContinuousCompiler._jax_binary(jax_lhs, jax_rhs, jax_op)
        
    def _jax_logical(self, expr, op, params):
        valid_ops = JaxRDDLContinuousCompiler.LOGICAL_OPS    
        JaxRDDLContinuousCompiler._check_valid_op(expr, valid_ops)     
        
        warnings.warn('Logical operator {} will be converted to arithmetic.'.format(
            valid_ops.keys()), FutureWarning, stacklevel=2)
        
        args = expr.args
        n = len(args)
        
        if n == 1 and op == '~':
            arg, = args
            jax_expr = self._jax(arg, params)
            return JaxRDDLContinuousCompiler._jax_unary(jax_expr, lambda e: 1.0 - e)
            
        elif n == 2:
            lhs, rhs = args
            jax_lhs = self._jax(lhs, params)
            jax_rhs = self._jax(rhs, params)
            jax_op = valid_ops[op]
            return JaxRDDLContinuousCompiler._jax_binary(jax_lhs, jax_rhs, jax_op)
        
        JaxRDDLContinuousCompiler._check_num_args(expr, 2)
    
    def _jax_aggregation(self, expr, op, params):
        valid_ops = JaxRDDLContinuousCompiler.AGGREGATION_OPS       
        JaxRDDLContinuousCompiler._check_valid_op(expr, valid_ops)    
        
        warnings.warn('Aggregation operator {} will be converted to arithmetic.'.format(
            valid_ops.keys()), FutureWarning, stacklevel=2)
            
        * pvars, arg = expr.args
        pvars = list(map(lambda p: p[1], pvars))  
        new_params = params + pvars
        reduced_axes = tuple(range(len(params), len(new_params)))
        
        jax_expr = self._jax(arg, new_params)
        jax_op = valid_ops[op]        
        
        def _f(x, key):
            val, key, err = jax_expr(x, key)
            sample = jax_op(val, axis=reduced_axes)
            return sample, key, err
        
        return _f

    def _jax_control(self, expr, op, params):
        valid_ops = {'if'}
        JaxRDDLContinuousCompiler._check_valid_op(expr, valid_ops)
        JaxRDDLContinuousCompiler._check_num_args(expr, 3)
        
        warnings.warn('Float predicate will be checked against 0.5.',
                      FutureWarning, stacklevel=2)
        
        pred, if_true, if_false = expr.args        
        jax_pred = self._jax(pred, params)
        jax_true = self._jax(if_true, params)
        jax_false = self._jax(if_false, params)
        
        def _f(x, key):
            val1, key, err1 = jax_pred(x, key)
            val1 = jnp.greater(val1, 0.5)
            val2, key, err2 = jax_true(x, key)
            val3, key, err3 = jax_false(x, key)
            sample = jnp.where(val1, val2, val3)
            err = err1 | err2 | err3
            return sample, key, err
            
        return _f

    def _jax_kron(self, expr, params):
        warnings.warn('KronDelta will be ignored.', FutureWarning, stacklevel=2)    
        
        arg, = expr.args
        arg = self._jax(arg, params)
        return arg
    
    def _jax_poisson(self, expr, params):
        raise RDDLNotImplementedError(
            'No reparameterization implemented for Poisson.' + '\n' + 
            JaxRDDLContinuousCompiler._print_stack_trace(expr))
    
    def _jax_gamma(self, expr, params):
        raise RDDLNotImplementedError(
            'No reparameterization implemented for Gamma.' + '\n' + 
            JaxRDDLContinuousCompiler._print_stack_trace(expr))
            
    
class JaxRDDLStraightlinePlanner:
    
    def __init__(self,
                 rddl: RDDL,
                 key: jax.random.PRNGKey,
                 n_steps: int,
                 n_batch: int,
                 initializer: jax.nn.initializers.Initializer=jax.nn.initializers.zeros,
                 optimizer: optax.GradientTransformation=optax.adam(0.1),
                 aggregation=jnp.mean) -> None:
        self.key = key
        
        compiler = JaxRDDLContinuousCompiler(rddl, allow_discrete=False)
        sim = JaxRDDLSimulator(compiler, key)
        subs = sim.subs
        cpfs = sim.cpfs
        reward_fn = sim.reward
        primed_unprimed = sim.state_unprimed
        
        NORMAL = JaxRDDLContinuousCompiler.ERROR_CODES['NORMAL']
        
        # plan initialization
        action_info = {}
        for pvar in rddl.domain.action_fluents.values():
            aname = pvar.name       
            atype = JaxRDDLContinuousCompiler.RDDL_TO_JAX_TYPE[pvar.range]
            ashape = (n_steps,)
            avalue = subs[aname]
            if hasattr(avalue, 'shape'):
                ashape = ashape + avalue.shape
            action_info[aname] = (atype, ashape)
            if atype != JaxRDDLContinuousCompiler.REAL:
                subs[aname] = np.asarray(subs[aname], dtype=np.float32)
                
        # perform one step of a roll-out        
        def _step(carry, actions):
            x, key, err = carry            
            x.update(actions)
            
            # calculate all the CPFs (e.g., s') and r(s, a, s') in order
            for name, cpf_fn in cpfs.items():
                x[name], key, cpf_err = cpf_fn(x, key)
                err |= cpf_err
            reward, key, rew_err = reward_fn(x, key)
            err |= rew_err
            
            # set s <- s'
            for primed, unprimed in primed_unprimed.items():
                x[unprimed] = x[primed]
                
            return (x, key, err), reward
        
        # perform a single roll-out
        def _rollout(plan, x0, key):
            
            # set s' <- s at the first epoch
            x0 = x0.copy()               
            for primed, unprimed in primed_unprimed.items():
                x0[primed] = x0[unprimed]
            
            # generate roll-outs and cumulative reward
            (x, key, err), rewards = jax.lax.scan(_step, (x0, key, NORMAL), plan)
            cuml_reward = jnp.sum(rewards)
            
            return cuml_reward, x, key, err
        
        # force action ranges         
        def _force_action_ranges(plan):
            new_plan = {}
            for name, value in plan.items():
                atype, _ = action_info[name]
                if atype == JaxRDDLContinuousCompiler.INT:
                    new_plan[name] = jnp.asarray(
                        value, dtype=JaxRDDLContinuousCompiler.REAL)
                elif atype == bool:
                    new_plan[name] = jax.nn.sigmoid(value)
                else:
                    new_plan[name] = value
            return new_plan
        
        self.force_action_ranges = jax.jit(_force_action_ranges)
        
        # do a batch of roll-outs
        def _batched(value):
            value = jnp.asarray(value)
            batched_shape = (n_batch,) + value.shape
            return jnp.broadcast_to(value, shape=batched_shape) 
        
        def _batched_rollouts(plan, key):
            x_batch = jax.tree_map(_batched, subs)
            keys = jax.random.split(key, num=n_batch)
            returns, x_batch, keys, errs = jax.vmap(
                _rollout, in_axes=(None, 0, 0))(plan, x_batch, keys)
            key = keys[-1]
            return returns, key, x_batch, errs
        
        # aggregate all sampled returns
        def _loss(plan, key):
            plan = _force_action_ranges(plan)
            returns, key, x_batch, errs = _batched_rollouts(plan, key)
            loss_value = -aggregation(returns)
            errs = jax.lax.reduce(errs, NORMAL, jnp.bitwise_or, (0,))
            return loss_value, (key, x_batch, errs)
        
        # gradient descent update
        def _update(plan, opt_state, key):
            grad, (key, x_batch, errs) = jax.grad(_loss, has_aux=True)(plan, key)
            updates, opt_state = optimizer.update(grad, opt_state)
            plan = optax.apply_updates(plan, updates)       
            return plan, opt_state, key, x_batch, errs
        
        self.loss = jax.jit(_loss)
        self.update = jax.jit(_update)
        
        # initialization
        def _initialize(key):
            plan = {}
            for action, (_, ashape) in action_info.items():
                key, subkey = random.split(key)
                plan[action] = initializer(
                    subkey, ashape, dtype=JaxRDDLContinuousCompiler.REAL)
            opt_state = optimizer.init(plan)
            return plan, opt_state, key
        
        self.initialize = jax.jit(_initialize)

    def optimize(self, n_epochs: int) -> Generator[Dict, None, None]:
        ''' Compute an optimal straight-line plan for the RDDL domain and instance.
        
        @param n_epochs: the maximum number of steps of gradient descent
        '''
        plan, opt_state, self.key = self.initialize(self.key)
        
        best_plan = self.force_action_ranges(plan)
        best_loss = float('inf')
        
        for step in range(n_epochs):
            plan, opt_state, self.key, _, _ = self.update(plan, opt_state, self.key)
                       
            loss_val, (self.key, rollouts, errs) = self.loss(plan, self.key)
            errs = JaxRDDLContinuousCompiler.get_error_codes(errs)
            fixed_plan = self.force_action_ranges(plan)

            if loss_val < best_loss:
                best_plan = fixed_plan
                best_loss = loss_val
            
            callback = {'step': step,
                        'plan': fixed_plan,
                        'best_plan': best_plan,
                        'loss': loss_val,
                        'best_loss': best_loss,
                        'rollouts': rollouts,
                        'errors': errs}
            yield callback
        
