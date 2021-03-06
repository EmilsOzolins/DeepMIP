import ecole
import numpy as np
import pyscipopt

from agents.primal_nn import NetworkPolicy, extract_current_ip_instance


class ObservationFunction():

    def __init__(self, problem):
        # called once for each problem benchmark
        self.problem = problem  # to devise problem-specific observations

    def seed(self, seed):
        # called before each episode
        # use this seed to make your code deterministic
        # return 0  # TODO: This is set just for comparison
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
        # return 0  # TODO: This is set just for comparison
        pass

    def before_reset(self, model):
        # called when a new episode is about to start
        pass

    def extract(self, model, done):
        m = model.as_pyscipopt()
        obj_val = m.getObjVal()
        sol = m.getBestSol()

        ip = extract_current_ip_instance(m)

        return sol, obj_val, ip


class SearchDynamics(ecole.dynamics.PrimalSearchDynamics):

    def __init__(self, trials_per_node=1, depth_freq=1, depth_start=0, depth_stop=-1):
        super().__init__(trials_per_node, depth_freq, depth_start, depth_stop)


class SCIPEnvironment(ecole.environment.PrimalSearch):
    __Dynamics__ = SearchDynamics

    def reset(self, instance, *dynamics_args, **dynamics_kwargs):
        self.can_transition = True
        try:
            self.model = ecole.core.scip.Model.from_pyscipopt(instance)
            # we want to focus on finding feasible solutions
            self.model.as_pyscipopt().setHeuristics(pyscipopt.scip.PY_SCIP_PARAMSETTING.AGGRESSIVE)
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
        self.network_policy = NetworkPolicy(problem)
        self.counter = 0
        self.step = 1

    def seed(self, seed):
        # called before each episode
        # use this seed to make your code deterministic
        self.rng = np.random.RandomState(seed)
        # return 0

    def __call__(self, action_vars_in, observation):
        model, m_isfresh, vars_orig = observation

        # reset solution improvement and solving time counters for freshly created models
        if m_isfresh:
            self.obs = None
            self.counter = 0
            self.step = 1
            m_orig = model.as_pyscipopt()
            remaining_time_budget = m_orig.getParam("limits/time") - m_orig.getSolvingTime()
            # print("remaining_time_budget", remaining_time_budget)
            model_copy = pyscipopt.Model(sourceModel=m_orig)
            model_copy.setPresolve(pyscipopt.scip.PY_SCIP_PARAMSETTING.OFF)  # presolve has already been done
            # we want to focus on finding feasible solutions
            model_copy.setHeuristics(pyscipopt.scip.PY_SCIP_PARAMSETTING.AGGRESSIVE)
            model_copy.setObjlimit(m_orig.getObjlimit())

            self.env = SCIPEnvironment(observation_function=ObservationFunction_inner())
            self.obs, self.action_set, _, self.done, info = self.env.reset(model_copy)
            self.m = self.env.model.as_pyscipopt()
            # we want to focus on finding feasible solutions
            self.m.setHeuristics(pyscipopt.scip.PY_SCIP_PARAMSETTING.AGGRESSIVE)

            # stop the agent before the environment times out
            self.m.setParam('limits/time', max(remaining_time_budget - 0.01, 0))
            self.reported_bound = self.env.model.primal_bound

        while self.env.model.primal_bound >= self.reported_bound and not self.done:
            if self.counter % self.step == 0:
                *_, ip = self.obs
                policy_action = self.network_policy(self.action_set, ip)
            else:
                policy_action = [], []
            self.counter += 1
            # print(len(self.action_set))
            self.obs, self.action_set, _, self.done, info = self.env.step(policy_action)

        # keep track of best sol improvements in the copied model to be able to set a limit

        # keep track of solving time already spedn in the model to be able to increment it appropriately
        self.solving_time_consumed = self.m.getSolvingTime()
        self.reported_bound = self.env.model.primal_bound

        if self.done:
            print(f"done in {self.m.getSolvingTime()} sec.")
            return [], []

        sol = self.m.getBestSol()
        primal_val = self.m.getPrimalbound()
        dual_val = self.m.getDualbound()

        print(
            f"{self.m.getSolvingTime()} seconds spend searching, best solution so far primal={primal_val} and dual={dual_val}")
        sol_vals = np.asarray([self.m.getSolVal(sol, var) for var in self.m.getVars(transformed=False)])
        action = (action_vars_in, sol_vals[action_vars_in])

        return action
