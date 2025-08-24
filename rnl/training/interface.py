from typing import List

from rnl.configs.config import (
    EnvConfig,
    RenderConfig,
    RobotConfig,
    SensorConfig,
    TrainerConfig,
)
from rnl.configs.rewards import RewardConfig
from rnl.training.learn import inference, training
from rnl.training.train import run_multiple_parallel_trainings

def robot(
    base_radius: float,
    max_vel_linear: float,
    max_vel_angular: float,
    wheel_distance: float,
    weight: float,
    threshold: float = 0.1,
    collision: float = 0.2,
    path_model: str =  "",
) -> RobotConfig:

    parameter = {
        "Base radius": base_radius,
        "Wheel distance": wheel_distance,
        "Weight": weight,
    }

    for name, value in parameter.items():
        if value <= 0:
            raise ValueError(f"Error: {name} must be greater than 0.")

    if threshold < 0:
        raise ValueError("Error: Threshold must be greater than or equal to 0.")
    if collision < 0:
        raise ValueError("Error: Collision must be greater than or equal to 0.")
    if path_model == "":
        path_model = "None"

    return RobotConfig(
        base_radius,
        max_vel_linear,
        max_vel_angular,
        wheel_distance,
        weight,
        threshold,
        collision,
        path_model,
    )

def sensor(
    fov: float = 270, num_rays: int = 10, min_range: float = 0.1, max_range: float = 10.0
) -> SensorConfig:
    if min_range < 0 or max_range < 0:
        raise ValueError(
            "Error: Minimum/Maximum range must be greater than or equal to 0."
        )
    if num_rays < 0:
        raise ValueError("Error: Number of rays must be greater than or equal to 0.")
    if fov < 0 or fov > 360:
        raise ValueError("Error: Field of view must be between 0 and 360 degrees.")
    if min_range > max_range:
        raise ValueError(
            "Error: Minimum range must be less than or equal to maximum range."
        )
    if min_range == max_range:
        raise ValueError("Error: Minimum range cannot be equal to maximum range.")
    if num_rays == 0:
        raise ValueError("Error: Number of rays cannot be equal to 0.")
    if fov == 0:
        raise ValueError("Error: Field of view cannot be equal to 0.")
    if num_rays > 10:
        raise ValueError("Error: Number of rays must be less than or equal to 10.")

    return SensorConfig(fov, num_rays, min_range, max_range)


def make(
    scalar: int = 1,
    folder_map: str = "None",
    name_map: str = "None",
    max_timestep: int = 1000,
) -> EnvConfig:

    if scalar < 0 or scalar > 100:
        raise ValueError("Error: Scalar must be between 0 and 100.")
    if max_timestep < 0:
        raise ValueError("Error: Maximum timestep must be greater than 0.")

    return EnvConfig(
        scalar=scalar,
        folder_map=folder_map,
        name_map=name_map,
        timestep=max_timestep,
    )


def render(controller: bool = False, debug: bool = True, plot: bool = False) -> RenderConfig:
    return RenderConfig(controller, debug, plot)


class Trainer:
    def __init__(
        self,
        robot_config: RobotConfig,
        sensor_config: SensorConfig,
        env_config: EnvConfig,
        render_config: RenderConfig,
    ):
        self.robot_config = robot_config
        self.sensor_config = sensor_config
        self.env_config = env_config
        self.render_config = render_config

    def learn(
        self,
        population: int = 2,
        loop_feedback: int = 10,
        description_task: str = "reach the goal without crashing",
        pretrained: str = "",
        use_llm: bool =  False,
        max_timestep_global: int = 5_000_000,
        seed: int = 1,
        batch_size: int = 64,
        hidden_size: int = 128,
        num_envs: int = 8,
        device: str = "cpu",
        checkpoint: int = 10000,
        checkpoint_path: str = "checkpoints",
        use_wandb: bool =  False,
        wandb_api_key: str = "",
        llm_api_key: str = "",
        lr: float = 1e-5,
        learn_step: int = 256,
        gae_lambda: float = 0.90,
        ent_coef: float = 0.05,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        update_epochs: int = 3,
        clip_range_vf: float = 0.2,
        target_kl: float = 0.025,
        name: str = "rnl",
        verbose: bool = False,
    ):

        if seed < 0:
            raise ValueError("Error: Seed must be greater than or equal to 0.")
        if max_timestep_global < 0:
            raise ValueError("Error: Maximum timestep must be greater than 0.")
        if batch_size < 0:
            raise ValueError("Error: Batch size must be greater than 0.")
        if num_envs < 0:
            raise ValueError("Error: Number of environments must be greater than 0.")
        if learn_step < 0:
            raise ValueError("Error: Learning step must be greater than 0.")
        if gae_lambda < 0:
            raise ValueError("Error: GAE lambda must be greater than 0.")
        if ent_coef < 0:
            raise ValueError("Error: Entropy coefficient must be greater than 0.")
        if vf_coef < 0:
            raise ValueError(
                "Error: Value function coefficient must be greater than 0."
            )
        if max_grad_norm < 0:
            raise ValueError("Error: Maximum gradient norm must be greater than 0.")
        if update_epochs < 0:
            raise ValueError("Error: Update epochs must be greater than 0.")
        if pretrained == "":
            pretrained = "None"

        trainer_config = TrainerConfig(
            pretrained=pretrained,
            use_llm=use_llm,
            max_timestep_global=max_timestep_global,
            seed=seed,
            batch_size=batch_size,
            hidden_size=hidden_size,
            num_envs=num_envs,
            device=device,
            checkpoint=checkpoint,
            checkpoint_path=checkpoint_path,
            use_wandb=use_wandb,
            wandb_api_key=wandb_api_key,
            wandb_mode="online",
            llm_api_key=llm_api_key,
            lr=lr,
            learn_step=learn_step,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            update_epochs=update_epochs,
            clip_range_vf=clip_range_vf,
            target_kl=target_kl,
            name=name,
            verbose=verbose,
        )

        if self.render_config.plot:
            raise ValueError("Error: Plot mode is not supported for training.")
        if self.render_config.controller:
            raise ValueError("Error: Controller mode is not supported for training.")
        if not self.render_config.debug:
            raise ValueError("Error: Debug mode needs to be enabled")

        reward_config = RewardConfig(
            params={
                "scale_orientation": 0.0,
                "scale_distance": 0.0,
                "scale_time": 0.02,
                "scale_obstacle": 0.01,
                "scale_angular": 0.005,
            },
        )

        if use_llm:
            config = {
                "robot_config": self.robot_config,
                "sensor_config": self.sensor_config,
                "env_config": self.env_config,
                "render_config": self.render_config,
                "trainer_config": trainer_config,
                "reward_config": reward_config,
            }
            configs = population * [config]
            run_multiple_parallel_trainings(
                num_loops=loop_feedback,
                initial_configs=configs,
                num_populations=population,
                description_task=description_task
            )
            return None

        else:
            metrics = training(
                self.robot_config,
                self.sensor_config,
                self.env_config,
                self.render_config,
                trainer_config,
                reward_config,
                print_parameter=True,
            )
            return metrics



class Simulation:
    def __init__(
        self,
        robot_config: RobotConfig,
        sensor_config: SensorConfig,
        env_config: EnvConfig,
        render_config: RenderConfig,
    ) -> None:
        self.robot_config = robot_config
        self.sensor_config = sensor_config
        self.env_config = env_config
        self.render_config = render_config

    def run(self) -> None:

        reward_config = RewardConfig(
            params={
                "scale_orientation": 0.0,
                "scale_distance": 0.0,
                "scale_time": 0.02,
                "scale_obstacle": 0.01,
                "scale_angular": 0.005,
            },
        )

        inference(
            self.robot_config,
            self.sensor_config,
            self.env_config,
            self.render_config,
            reward_config,
        )

        return None
