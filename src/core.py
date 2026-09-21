
from src.data_models import State
from src.geometry import SplineTrajectorySE3
from src.sensors import Sensor
from src.estimators import Estimator


class Simulator:
    def __init__(self, trajectory: SplineTrajectorySE3, dt: float = 0.001):
        self.trajectory = trajectory
        self.dt = dt

        self.sensors: list[Sensor] = []
        self.estimators: list[Estimator] = []

    def add_sensor(self, sensor: Sensor):
        sensor.attach_simulation_dt(self.dt)
        self.sensors.append(sensor)

    def add_estimator(self, estimator: Estimator | list[Estimator]):
        if isinstance(estimator, list):
            for est in estimator: self.add_estimator(est)

        elif isinstance(estimator, Estimator):
            self.estimators.append(estimator)

        else:
            raise TypeError(f"estimator is not an instance of Estimator : {type(estimator)}")

    def run(self, t_start: float, t_end: float):
        step = 0
        current_time = t_start

        ### Initialization ###

        initial_gt_state = self.trajectory.get_state(current_time)

        # Initialize estimators if needed
        for estimator in self.estimators:
            estimator.initialize(initial_gt_state)

        # Register initial conditions (Step 0) in history
        gt_history: list[State] = [initial_gt_state]
        est_history: dict[int, list[State]] = {}

        for estimator in self.estimators:
            est_state = estimator.get_estimated_state()
            if est_state is not None:
                est_history[id(estimator)] = [est_state]
            else:
                est_history[id(estimator)] = []

        step = 1
        current_time = t_start + step * self.dt


        ### Loop ###

        while current_time <= t_end:
            # obtain ground truth
            gt_state = self.trajectory.get_state(current_time)
            gt_history.append(gt_state)

            # (Optional) estimator prediction
            for estimator in self.estimators:
                estimator.predict(self.dt)

            # sensor sampling
            for sensor in self.sensors:
                if sensor.should_sample(step):
                    meas = sensor.measure(gt_state)

                    # feed estimators with measurements
                    for estimator in self.estimators:
                        estimator.update(meas)

            # register estimations
            for estimator in self.estimators:
                est_state = estimator.get_estimated_state()
                if est_state:
                    est_history[id(estimator)].append(est_state)

            # advance simulation step
            step += 1
            current_time = t_start + step * self.dt

        return gt_history, est_history