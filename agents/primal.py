import ecole
import numpy as np
import pyscipopt


class ObservationFunction():

    def __init__(self, problem):
        # called once for each problem benchmark
        self.problem = problem  # to devise problem-specific observations
        self.isFresh = None

    def seed(self, seed):
        # called before each episode
        # use this seed to make your code deterministic
        pass

    def before_reset(self, model):
        # called when a new episode is about to start
        self.isFresh = True

    def extract(self, model, done):
        if done:
            return None

        m = model.as_pyscipopt()
        variables = m.getVars(transformed=True)

        observation = (model, self.isFresh, variables)
        self.isFresh = False
        return observation


class ObservationFunction_inner():

    def __init__(self):
        # called once for each problem benchmark
        pass

    def seed(self, seed):
        # called before each episode
        # use this seed to make your code deterministic
        pass

    def before_reset(self, model):
        # called when a new episode is about to start
        pass

    def extract(self, model, done):
        if done:
            return None

        m = model.as_pyscipopt()
        obj_val = m.getObjVal()
        sol = m.getBestSol()

        observation = (sol, obj_val)
        return observation


class SearchDynamics(ecole.dynamics.PrimalSearchDynamics):

    def __init__(self, trials_per_node=1, depth_freq=1, depth_start=0, depth_stop=-1):
        super().__init__(trials_per_node, depth_freq, depth_start, depth_stop)


class SCIPEnvironment(ecole.environment.PrimalSearch):
    __Dynamics__ = SearchDynamics

    def reset(self, instance, *dynamics_args, **dynamics_kwargs):
        self.can_transition = True
        try:
            self.model = ecole.core.scip.Model.from_pyscipopt(instance)
            self.model.as_pyscipopt().setHeuristics(
                pyscipopt.scip.PY_SCIP_PARAMSETTING.AGGRESSIVE)  # we want to focus on finding feasible solutions
            self.model.set_params(self.scip_params)
            self.model.disable_presolve()

            self.dynamics.set_dynamics_random_state(self.model, self.random_engine)

            # Reset data extraction functions
            self.reward_function.before_reset(self.model)
            self.observation_function.before_reset(self.model)
            self.information_function.before_reset(self.model)

            # Place the environment in its initial state
            done, action_set = self.dynamics.reset_dynamics(
                self.model, *dynamics_args, **dynamics_kwargs
            )
            self.can_transition = not done

            # Extract additional information to be returned by reset
            reward_offset = self.reward_function.extract(self.model, done)
            if not done:
                observation = self.observation_function.extract(self.model, done)
            else:
                observation = None
            information = self.information_function.extract(self.model, done)

            return observation, action_set, reward_offset, done, information
        except Exception as e:
            self.can_transition = False
            raise e


class Policy():

    def __init__(self, problem):
        # called once for each problem benchmark
        self.rng = np.random.RandomState()
        self.problem = problem  # to devise problem-specific policies
        self.env = None
        self.m = None

    def seed(self, seed):
        # called before each episode
        # use this seed to make your code deterministic
        self.rng = np.random.RandomState(seed)

    def __call__(self, action_vars_in, observation):
        model, m_isfresh, vars_orig = observation

        # reset solution improvement and solving time counters for freshly created models
        if m_isfresh:
            m_orig = model.as_pyscipopt()
            remaining_time_budget = m_orig.getParam("limits/time") - m_orig.getSolvingTime()
            # print("remaining_time_budget", remaining_time_budget)
            model_copy = pyscipopt.Model(sourceModel=m_orig)
            model_copy.setPresolve(pyscipopt.scip.PY_SCIP_PARAMSETTING.OFF)  # presolve has already been done
            # we want to focus on finding feasible solutions
            model_copy.setHeuristics(pyscipopt.scip.PY_SCIP_PARAMSETTING.AGGRESSIVE)

            env = SCIPEnvironment(observation_function=ObservationFunction_inner())
            obs, self.action_set, _, self.done, info = env.reset(model_copy)
            self.m = env.model.as_pyscipopt()
            # we want to focus on finding feasible solutions
            self.m.setHeuristics(pyscipopt.scip.PY_SCIP_PARAMSETTING.AGGRESSIVE)

            # stop the agent before the environment times out
            self.m.setParam('limits/time', max(remaining_time_budget - 0.01, 0))
            self.env = env
            self.reported_bound = self.env.model.primal_bound

        while self.env.model.primal_bound >= self.reported_bound and not self.done:
            policy_action = ([], [])  # todo our policy
            # print(len(self.action_set))
            obs, self.action_set, _, self.done, info = self.env.step(policy_action)

        # keep track of best sol improvements in the copied model to be able to set a limit

        # keep track of solving time already spedn in the model to be able to increment it appropriately
        self.solving_time_consumed = self.m.getSolvingTime()
        self.reported_bound = self.env.model.primal_bound

        if self.done:
            print(f"done in {self.m.getSolvingTime()} sec.")
            return [], []

        sol, obj_val = obs
        # print('nSols', self.m.getNSols())
        print(f"{self.m.getSolvingTime()} seconds spend searching, best solution so far {obj_val}")
        # sol_vals = np.asarray([sol[var] for var in vars_orig])
        sol_vals = np.asarray([self.m.getSolVal(sol, var) for var in self.m.getVars(transformed=False)])
        action = (action_vars_in, sol_vals[action_vars_in])

        return action
