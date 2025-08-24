import argparse
import os
import rnl as vault

def main(arg):
    wandb_key = os.environ.get("WANDB_API_KEY")
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    # 1.step -> config robot
    param_robot = vault.robot(
        base_radius=0.105,
        max_vel_linear=0.22,
        max_vel_angular=2.84,
        wheel_distance=0.16,
        weight=1.0,
        threshold=0.1,
        collision=0.05,
        path_model="",
    )

    # 2.step -> config sensors [for now only lidar sensor!!]
    param_sensor = vault.sensor(
        fov=270,
        num_rays=5,
        min_range=0.0,
        max_range=3.5,
    )

    # 3.step -> config env
    param_env = vault.make(
        scalar=12,
        folder_map="/Users/nicolasalan/Documents/rnl/map",
        name_map="map",
        max_timestep=1000,
    )
    # 4.step -> config render
    param_render = vault.render(controller=True, debug=True, plot=False)

    if args.mode == "learn":
        # 5.step -> config train robot
        model = vault.Trainer(
            param_robot,
            param_sensor,
            param_env,
            param_render,
        )

        # 6.step -> train robot
        model.learn(
            population=0,
            loop_feedback=0,
            description_task="reach the goal without crashing",
            pretrained="",
            use_llm=False,
            max_timestep_global=5_000_000,
            seed=1,
            batch_size=1024,
            hidden_size=128,
            num_envs=16,
            device="mps",
            checkpoint=10000,
            checkpoint_path="checkpoints",
            use_wandb=False,
            wandb_api_key=str(wandb_key),
            llm_api_key=str(gemini_api_key),
            lr=1e-5,
            learn_step=512,
            gae_lambda=0.95,
            ent_coef=0.05,
            vf_coef=0.5,
            max_grad_norm=0.5,
            update_epochs=8,
            clip_range_vf=0.2,
            target_kl=0.025,
            name="rnl",
            verbose=True,
        )

    elif args.mode == "sim":
        # 5.step -> config train robot
        model = vault.Simulation(param_robot, param_sensor, param_env, param_render)
        # 6.step -> run robot
        model.run()


if __name__ == "__main__":
    print("run ...")
    parser = argparse.ArgumentParser(description="Train or setup environment.")
    parser.add_argument("mode", choices=["learn", "sim"], help="Mode")

    args = parser.parse_args()
    main(args)
